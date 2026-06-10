#!/usr/bin/env python3
"""Convert GH Archive issue/PR event streams to JEPA chains (issue lifecycles).

WORLD-STATE READING (worldmodel_chops_data.md rank 1 — the "world-model chops" tier):
A GitHub issue is a real-world ENTITY with a mechanical state machine. Each GH Archive
event (https://gharchive.org) carries the FULL issue snapshot embedded in its payload,
so we do not reconstruct state from a state machine — we READ it directly off each event:

    state  = (issue_state, labels, assignee, milestone, locked) rendered as short text
    action = the real GitHub event that produced the *next* state
             (opened / closed / reopened / labeled / unlabeled / assigned /
              unassigned / milestoned / demilestoned / commented / ...)

A CHAIN is one issue's chronological event sequence (capped at --max-events states):

    {"chain": [s0_text, s1_text, ..., sk_text]}                    # plain
    {"chain": [...], "actions": ["<action>@0", ...], ...}          # labeled twin

This is the closest real-world analog to the repo's entity-world (data/entity_world):
one entity per chain, explicit per-transition actions, deterministic-given-action
dynamics. The existing JEPA instruments (action-NMI, target-recovery — diagnostics.py
§3a/§3c, which read `actions` as "<verb>@<idx>" and strip @idx for verb-only NMI) work
UNMODIFIED: the single issue is entity slot 0, so every action is tagged "@0".

------------------------------------------------------------------------------
WHY THIS TESTS WORLD-MODEL CAPABILITIES (not language scale)
------------------------------------------------------------------------------
  - STATE TRACKING: the rendered state is a small structured attribute set
    (status/labels/assignee/milestone). Predicting state_{t+1} requires carrying the
    label/assignee SET forward, not paraphrasing prose.
  - ACTION-CONDITIONED PREDICTION: the action ("labeled bug") deterministically
    dictates the next-state delta (the bug label appears). Transition determinism
    given the action is HIGH — measured ~98%+ on the held-out sample (see the
    converter's --report mode and worldmodel_chops_data.md).
  - OOD ENTITIES: issues/repos created AFTER a cutoff date are unseen test entities.
    The converter's iid split is the in-distribution baseline; a date-held-out split
    (server-side, two non-overlapping day ranges) gives the OOD-entity probe.
  - CALIBRATION: action frequency given a state (which event tends to follow an open,
    labeled, assigned issue?) has empirical ground-truth frequencies over the corpus.

------------------------------------------------------------------------------
STATE RENDERING (deterministic, short, attribute-set form)
------------------------------------------------------------------------------
render_state() turns an issue snapshot into a canonical sentence-set so the BPE/decoder
sees structure, not prose:

    "issue 4312 is open. it has label bug. it has label help wanted.
     it is assigned to alice. it has milestone v2. "

Labels are sorted (set-stable: order must NOT carry signal). Assignee/milestone are
included only when present. Comment/title bodies are DROPPED — we keep the mechanical
state, not the discussion text (that would reintroduce the language-scale confound the
survey warns about). The issue number anchors the entity across the chain.

------------------------------------------------------------------------------
ACTION EXTRACTION (explicit; from the event, not inferred)
------------------------------------------------------------------------------
extract_action() reads the GH event into a short action token:
  - IssuesEvent/PullRequestEvent action -> "opened"/"closed"/"reopened"/"locked"/...
  - "labeled"/"unlabeled": the payload's `label` names the specific label, so we emit
    "labeled" (the WHICH-label is recoverable from the state diff, kept coarse for the
    action vocab). Same for "assigned"/"unassigned"/"milestoned".
  - IssueCommentEvent -> "commented".
The action for transition i is the action of the event that PRODUCED state_{i+1}
(entity-world convention: the action drives the next state). Tagged "@0" (single entity).

------------------------------------------------------------------------------
SCALE: local sample vs full server run
------------------------------------------------------------------------------
GH Archive ships one gzipped JSON file per hour (~120 MB, ~290K events). Chains accrue
across hours: ~9 hours of one day yield ~14K len>=2 / ~5K len>=3 issue chains (measured).
--inputs takes a glob of local hourly .json.gz files (download them with the printed
server command). The SAME command runs locally on a downloaded sample and server-side on
a full month. --report prints transition-determinism + action-vocab stats and exits.

Usage (local sample, after downloading ~9 hourly files to data/gharchive/raw/):
    uv run python scripts/convert_gharchive.py \
        --inputs 'data/gharchive/raw/2024-01-15-*.json.gz' \
        --out-dir data/gharchive --n-train 4000 --n-test 500 \
        --min-events 2 --max-events 12 --seed 7

Usage (full, server-side — fetch a month, then convert):
    for d in $(seq -w 1 31); do for h in $(seq -w 0 23); do \
        curl -sSL -o data/gharchive/raw/2024-01-$d-$h.json.gz \
          https://data.gharchive.org/2024-01-$d-$h.json.gz; done; done
    uv run python scripts/convert_gharchive.py \
        --inputs 'data/gharchive/raw/2024-01-*.json.gz' \
        --out-dir data/gharchive --n-train 200000 --n-test 5000 --seed 7
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Action vocabulary (explicit GH event actions). Kept coarse so the action codebook
# is small and the verb-NMI is meaningful; the WHICH-label/WHICH-assignee detail is
# recoverable from the state diff, not the action token.
# ---------------------------------------------------------------------------
ISSUE_ACTIONS = {
    "opened", "closed", "reopened", "labeled", "unlabeled",
    "assigned", "unassigned", "milestoned", "demilestoned",
    "locked", "unlocked", "transferred", "pinned", "unpinned",
    "edited", "deleted",
}

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Lowercase + whitespace-collapse a label/login/title fragment."""
    return _WS.sub(" ", (text or "").strip().lower())


def render_state(issue: dict) -> str:
    """Render an issue snapshot to a canonical short attribute-set sentence string.

    Deterministic and ORDER-STABLE (labels sorted) so label order carries no signal.
    Only mechanical state is kept (status / labels / assignee / milestone / locked);
    free-text title/body are dropped to keep this a world-state corpus, not prose.
    """
    num = issue.get("number")
    parts: list[str] = []
    status = _norm(issue.get("state") or "open")
    parts.append(f"issue {num} is {status}.")
    labels = sorted({_norm(l.get("name", "")) for l in (issue.get("labels") or []) if l.get("name")})
    for lab in labels:
        parts.append(f"it has label {lab}.")
    # assignee: prefer the assignees list (multi), fall back to single assignee.
    assignees = issue.get("assignees")
    logins: list[str] = []
    if assignees:
        logins = sorted({_norm(a.get("login", "")) for a in assignees if a.get("login")})
    elif issue.get("assignee"):
        logins = [_norm(issue["assignee"].get("login", ""))]
    for lg in logins:
        if lg:
            parts.append(f"it is assigned to {lg}.")
    if not logins:
        parts.append("it is unassigned.")
    ms = issue.get("milestone")
    if ms and ms.get("title"):
        parts.append(f"it has milestone {_norm(ms['title'])}.")
    if issue.get("locked"):
        parts.append("it is locked.")
    return " ".join(parts)


def extract_action(event_type: str, payload: dict) -> str:
    """Map a GH event to a coarse action token (the action that produced the new state)."""
    if event_type == "IssueCommentEvent":
        return "commented"
    action = _norm(payload.get("action") or "")
    if action in ISSUE_ACTIONS:
        return action
    # PullRequestEvent uses synchronize/ready_for_review etc.; fold rare ones to "edited".
    if action in ("synchronize", "ready_for_review", "review_requested", "auto_merge_enabled"):
        return "edited"
    return action or "other"


# ---------------------------------------------------------------------------
# Streaming event reader: group issue/PR events by (repo, number) -> chronological chain
# ---------------------------------------------------------------------------

def _iter_events(paths: list[Path]):
    """Yield parsed events from gzipped (or plain) GH Archive hourly JSON files."""
    for p in paths:
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt", errors="replace") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue


def collect_issue_events(
    paths: list[Path],
    include_prs: bool,
) -> dict[tuple, list[dict]]:
    """Group issue-touching events by (repo, number).

    Returns {key: [event_record, ...]} where each event_record is
    {"ts": str, "type": str, "action": str, "issue": snapshot_dict}. Order within a key
    is the file/stream order; we sort by ts before chaining. PRs are included via
    PullRequestEvent (payload.pull_request is an issue-shaped snapshot) when include_prs.
    """
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for e in _iter_events(paths):
        t = e.get("type")
        if t == "IssuesEvent" or t == "IssueCommentEvent":
            p = e.get("payload", {})
            iss = p.get("issue")
            if not iss or iss.get("pull_request"):  # skip PR-comment leakage into issues
                continue
            kind = "issue"
        elif t == "PullRequestEvent" and include_prs:
            p = e.get("payload", {})
            iss = p.get("pull_request")
            if not iss:
                continue
            kind = "pr"
        else:
            continue
        num = iss.get("number")
        if num is None:
            continue
        repo = (e.get("repo") or {}).get("name", "?")
        key = (repo, num, kind)
        groups[key].append({
            "ts": e.get("created_at") or "",
            "type": t,
            "action": extract_action(t, p),
            "issue": iss,
        })
    return groups


def build_chains(
    groups: dict[tuple, list[dict]],
    min_events: int,
    max_events: int,
) -> tuple[list[dict], list[dict], dict]:
    """Turn grouped events into (plain_chains, labeled_chains, stats).

    A chain is one entity's chronological states. The action for transition i is the
    action of the event that produced state_{i+1} (entity-world convention).
    Consecutive duplicate states (an event that didn't change the rendered state, e.g.
    a comment) are KEPT — the action ("commented") is still a real, learnable no-op
    transition, mirroring entity-world's "wait"/"time passes" identity steps.
    """
    plain: list[dict] = []
    labeled: list[dict] = []
    n_too_short = 0
    n_truncated = 0
    event_counts: list[int] = []

    for (repo, num, kind), evs in groups.items():
        evs = sorted(evs, key=lambda r: r["ts"])
        if len(evs) < min_events:
            n_too_short += 1
            continue
        if len(evs) > max_events:
            evs = evs[:max_events]
            n_truncated += 1
        states = [render_state(ev["issue"]) for ev in evs]
        # action for transition i (s_i -> s_{i+1}) is the action of event i+1.
        actions = [f"{evs[i + 1]['action']}@0" for i in range(len(evs) - 1)]
        plain.append({"chain": states})
        labeled.append({
            "chain": states,
            "actions": actions,
            "types": [kind],            # single entity-type slot (issue|pr) per chain
            "repo": repo,
            "number": num,
        })
        event_counts.append(len(states))

    stats = {
        "issues_grouped": len(groups),
        "chains_kept": len(plain),
        "dropped_too_short": n_too_short,
        "truncated_over_max": n_truncated,
        "min_events": min_events,
        "max_events": max_events,
        "mean_events": round(sum(event_counts) / len(event_counts), 2) if event_counts else 0,
        "transitions": sum(c - 1 for c in event_counts),
    }
    return plain, labeled, stats


# ---------------------------------------------------------------------------
# Determinism / calibration report (the "world-model chops" measurement)
# ---------------------------------------------------------------------------

def determinism_report(labeled: list[dict]) -> dict:
    """Measure transition determinism given the action + action-vocab + state-cond entropy.

    - det_given_action: for each (state_text, action) seen >=THRESH times, the fraction of
      the modal next-state. Mean over keys = P(correct next-state | state, action) ceiling.
      HIGH means the state machine is mechanical (a good world-model testbed).
    - H(action | state): the stochastic part — which event tends to follow a given state.
    """
    THRESH = 5
    sa_next: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    s_act: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    action_vocab: collections.Counter = collections.Counter()

    for rec in labeled:
        chain, actions = rec["chain"], rec["actions"]
        for i in range(len(actions)):
            s, a, ns = chain[i], actions[i], chain[i + 1]
            verb = a.split("@")[0]
            action_vocab[verb] += 1
            sa_next[(s, verb)][ns] += 1
            s_act[s][verb] += 1

    det = [c.most_common(1)[0][1] / sum(c.values())
           for c in sa_next.values() if sum(c.values()) >= THRESH]

    def entropy(c: collections.Counter) -> float:
        tot = sum(c.values())
        return -sum((v / tot) * math.log2(v / tot) for v in c.values()) if tot else 0.0

    s_ent = [entropy(c) for c in s_act.values() if sum(c.values()) >= THRESH]
    return {
        "action_vocab_size": len(action_vocab),
        "action_vocab": dict(action_vocab.most_common()),
        "det_given_action_mean": round(sum(det) / len(det), 4) if det else None,
        "det_keys_counted": len(det),
        "H_action_given_state_mean_bits": round(sum(s_ent) / len(s_ent), 4) if s_ent else None,
        "state_keys_counted": len(s_ent),
        "threshold": THRESH,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="glob of GH Archive hourly .json.gz files")
    ap.add_argument("--out-dir", default="data/gharchive")
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--n-test", type=int, default=5_000)
    ap.add_argument("--min-events", type=int, default=2, help="min events to keep an issue as a chain")
    ap.add_argument("--max-events", type=int, default=12, help="truncate chains longer than this")
    ap.add_argument("--no-prs", action="store_true", help="issues only (drop PullRequestEvent chains)")
    ap.add_argument("--report", action="store_true", help="print determinism/calibration report and exit")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    paths = [Path(p) for p in sorted(glob.glob(args.inputs))]
    if not paths:
        raise SystemExit(f"No files match --inputs {args.inputs!r}. Download GH Archive hourly dumps first.")
    print(f"Source: GH Archive (gharchive.org) — {len(paths)} hourly files")
    print(f"  first: {paths[0].name}  last: {paths[-1].name}")
    print(f"Streaming + grouping issue/PR events (include_prs={not args.no_prs})...")

    groups = collect_issue_events(paths, include_prs=not args.no_prs)
    plain, labeled, stats = build_chains(groups, args.min_events, args.max_events)
    print(f"Grouped {stats['issues_grouped']} entities -> {len(plain)} chains "
          f"(dropped {stats['dropped_too_short']} too-short, truncated {stats['truncated_over_max']}). "
          f"~{stats['transitions']} transitions, mean {stats['mean_events']} events/chain.")

    report = determinism_report(labeled)
    print(f"\n=== World-model chops report ===")
    print(f"  action vocab ({report['action_vocab_size']}): {report['action_vocab']}")
    print(f"  determinism given action (mean modal-next fraction over {report['det_keys_counted']} keys): "
          f"{report['det_given_action_mean']}  [HIGH => mechanical state machine]")
    print(f"  H(action | state) over {report['state_keys_counted']} keys: "
          f"{report['H_action_given_state_mean_bits']} bits  [the stochastic part]")
    if args.report:
        return

    rng = random.Random(args.seed)
    order = list(range(len(plain)))
    rng.shuffle(order)
    plain = [plain[i] for i in order]
    labeled = [labeled[i] for i in order]

    n_test = min(args.n_test, len(plain))
    test_plain, test_labeled = plain[:n_test], labeled[:n_test]
    rest_plain, rest_labeled = plain[n_test:], labeled[n_test:]
    n_train = min(args.n_train, len(rest_plain))
    train_plain, train_labeled = rest_plain[:n_train], rest_labeled[:n_train]

    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", train_plain)
    write_jsonl(out / "train_labeled.jsonl", train_labeled)
    write_jsonl(out / "test.jsonl", test_plain)
    write_jsonl(out / "test_labeled.jsonl", test_labeled)

    manifest = {
        "corpus": "gharchive",
        "source": "https://gharchive.org (hourly GH event dumps)",
        "license": "GH Archive is a public dataset of GitHub's public event stream; "
                   "GitHub Terms of Service apply. Free for research.",
        "world_state_reading": "entity=issue/PR; state=(status,labels,assignee,milestone,locked) "
                               "rendered text; action=real GH event (opened/closed/labeled/...)",
        "action_extraction": "coarse GH event action token; @0 single-entity slot",
        "state_rendering": "canonical sorted attribute-set sentences (mechanical state only, no prose)",
        "include_prs": not args.no_prs,
        "seed": args.seed,
        "min_events": args.min_events,
        "max_events": args.max_events,
        "report": report,
        "splits": {
            "train": {"chains": len(train_plain),
                      "transitions": sum(len(c["chain"]) - 1 for c in train_plain)},
            "test": {"chains": len(test_plain),
                     "transitions": sum(len(c["chain"]) - 1 for c in test_plain)},
        },
        "build_stats": stats,
        "files": ["train.jsonl", "train_labeled.jsonl", "test.jsonl", "test_labeled.jsonl"],
        "ood_note": "for an OOD-entity probe, convert two non-overlapping date ranges and "
                    "use the later range as test (issues/repos created after the cutoff).",
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote to {out}/:")
    print(f"  train.jsonl          {len(train_plain):>7} chains")
    print(f"  train_labeled.jsonl  {len(train_labeled):>7} chains")
    print(f"  test.jsonl           {len(test_plain):>7} chains")
    print(f"  test_labeled.jsonl   {len(test_labeled):>7} chains")
    print(f"  manifest.json")
    if train_labeled:
        print("\nSample chain:")
        ex = train_labeled[0]
        print(f"  repo {ex['repo']} #{ex['number']} ({ex['types'][0]})")
        for j, st in enumerate(ex["chain"]):
            act = f"  --[{ex['actions'][j]}]-->" if j < len(ex["actions"]) else ""
            print(f"    s{j}: {st[:90]}{act}")


if __name__ == "__main__":
    main()
