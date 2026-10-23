"""Student models that copy Hearthvale's decision model.

Both read a question as triples and score each offered option; the answer is
a softmax over the offered options, trained to match the teacher's
distribution (KL(teacher || student)).

``DecisionSetModel`` (the triples shape)
    Token = scope + attribute + value embeddings, plus ``num · num_dir[attr]``
    for numeric values. No positions: the triples are a set.
    State triples (town, district, proposal, body) go through a small
    self-attention encoder. That memory depends only on the snapshot, so the
    game encodes it once per state (a season) and caches it.
    Question triples (kind, context) and one token per offered option then run
    through decoder layers: self-attention among themselves plus cross-attention
    into the state memory. Each option token ends in one logit.

``DecisionTWMModel`` (the TWM core's token layout)
    As the set model, but each triple is three tokens, as in TWM's
    TripleEncoder: scope, attribute and value each from their own table, plus a
    role embedding (entity, relation, value: always that order) and the triple's
    position (``triple_pos_emb``). Numbers ride on the value token. Attention can
    then relate a value to other triples' parts, not only whole triples.

``DecisionMLP`` (the baseline: Deep Sets)
    Each triple is read on its own first: its scope, attribute and value
    embeddings (the value carrying any number) side by side, through a small
    per-triple network φ; the triples' outputs are summed and an MLP ρ reads the
    sum; option logits are dot products with option embeddings. Concatenating
    the parts keeps which value belongs to which attribute: summing every
    triple's scope + attribute + value directly would reduce a question to three
    bags (scopes, attributes, values) and lose that binding. Train it on one
    kind (``--kinds resident_routine``) for a per-task model, or on all kinds.

``DecisionFM`` (a factorization machine, multi-output)
    Every pair of triples interacts through the elementwise product of their
    embeddings; each option weights those interactions its own way:
    logit(o) = <u_o, LN(½((Σe)² − Σe²)) + LN(Σe)>. O(triples · d), no hidden
    layers: generalizes to unseen pairs whose halves were seen.
    ``fm_order=3`` adds every triple of triples, from a projection f = eW of the
    embeddings: Σ_{i<j<k} f_i f_j f_k = (p1³ − 3·p1·p2 + 2·p3) / 6 with
    p_m = Σ f^m (Newton's identities), still O(triples · d).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DecisionVocab


@dataclass
class StudentConfig:
    arch: str = "set"          # 'set' | 'twm' | 'mlp' | 'fm'
    d_model: int = 32
    n_heads: int = 2
    state_layers: int = 2
    question_layers: int = 2
    d_ff: int = 64
    mlp_hidden: int = 128
    dropout: float = 0.0
    max_triples: int = 96      # 'twm': triple positions
    fm_order: int = 2          # 'fm': 2 (pairs) or 3 (pairs and triples)

    def to_dict(self) -> dict:
        return asdict(self)


class TripleEmbedding(nn.Module):
    def __init__(self, vocab: DecisionVocab, d: int):
        super().__init__()
        self.scope = nn.Embedding(len(vocab.scopes), d, padding_idx=0)
        self.attr = nn.Embedding(len(vocab.attrs), d, padding_idx=0)
        self.value = nn.Embedding(len(vocab.values), d, padding_idx=0)
        # Direction a numeric value moves along, per attribute.
        self.num_dir = nn.Embedding(len(vocab.attrs), d, padding_idx=0)
        self.kind_opt = nn.Embedding(len(vocab.kind_options), d, padding_idx=0)
        self.opt = nn.Embedding(len(vocab.options), d, padding_idx=0)
        for emb in (self.scope, self.attr, self.value, self.num_dir, self.kind_opt, self.opt):
            nn.init.normal_(emb.weight, std=0.1)
            emb.weight.data[0].zero_()

    def triples(self, b: dict) -> torch.Tensor:
        x = self.scope(b["scope"]) + self.attr(b["attr"]) + self.value(b["value"])
        return x + (b["num"] * b["has_num"]).unsqueeze(-1) * self.num_dir(b["attr"])

    def triple_parts(self, b: dict) -> torch.Tensor:
        """Each triple's scope, attribute and value (with its number) side by side: (B, T, 3d)."""
        value = self.value(b["value"]) + (b["num"] * b["has_num"]).unsqueeze(-1) * self.num_dir(b["attr"])
        return torch.cat([self.scope(b["scope"]), self.attr(b["attr"]), value], dim=-1)

    def options(self, b: dict) -> torch.Tensor:
        return self.kind_opt(b["kind_opt"]) + self.opt(b["opt"])


class DecisionSetModel(nn.Module):
    def __init__(self, vocab: DecisionVocab, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = TripleEmbedding(vocab, d)
        self.register_buffer("state_scopes", torch.tensor(vocab.state_scope_ids or [0], dtype=torch.long), persistent=False)
        self.option_flag = nn.Parameter(torch.zeros(d))  # marks option tokens apart from question triples
        # With no state layers the snapshot's facts reach the decision layer as their
        # embeddings alone (normalised): the per-snapshot cache is then a lookup.
        self.state_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout, batch_first=True, norm_first=True),
            cfg.state_layers, enable_nested_tensor=False) if cfg.state_layers > 0 else None
        self.state_norm = nn.LayerNorm(d)
        self.question = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout, batch_first=True, norm_first=True),
            cfg.question_layers)
        self.out_norm = nn.LayerNorm(d)
        self.score = nn.Linear(d, 1)

    def tokens(self, b: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (tokens (B, N, d), real (B, N), part of the snapshot (B, N)): one token per triple."""
        return self.embed.triples(b), b["tok_mask"], b["tok_mask"] & torch.isin(b["scope"], self.state_scopes)

    def encode_state(self, x: torch.Tensor, is_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """State memory (B, N, d) and its padding mask (True = ignore)."""
        # A question with no snapshot still needs one key to attend to.
        empty = ~is_state.any(dim=1)
        is_state = is_state.clone()
        is_state[empty, 0] = True
        mem = self.state_encoder(x, src_key_padding_mask=~is_state) if self.state_encoder is not None else x
        return self.state_norm(mem), ~is_state

    def forward(self, b: dict, return_hidden: bool = False):
        q, real, is_state = self.tokens(b)
        mem, mem_pad = self.encode_state(q, is_state)
        is_q = real & ~is_state
        o = self.embed.options(b) + self.option_flag
        tgt = torch.cat([q, o], dim=1)
        tgt_pad = ~torch.cat([is_q, b["opt_mask"]], dim=1)
        h = self.question(tgt, mem, tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=mem_pad)
        hidden = self.out_norm(h[:, q.shape[1]:])
        logits = self.score(hidden).squeeze(-1).masked_fill(~b["opt_mask"], float("-inf"))
        # The option tokens' final states, for the confidence head (confidence.py).
        return (logits, hidden) if return_hidden else logits


class DecisionTWMModel(DecisionSetModel):
    def __init__(self, vocab: DecisionVocab, cfg: StudentConfig):
        super().__init__(vocab, cfg)
        self.role = nn.Embedding(3, cfg.d_model)
        self.triple_pos = nn.Embedding(cfg.max_triples, cfg.d_model)
        nn.init.normal_(self.role.weight, std=0.1)
        nn.init.normal_(self.triple_pos.weight, std=0.1)

    def tokens(self, b: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = self.embed
        value = e.value(b["value"]) + (b["num"] * b["has_num"]).unsqueeze(-1) * e.num_dir(b["attr"])
        x = torch.stack([e.scope(b["scope"]), e.attr(b["attr"]), value], dim=2)        # (B, T, 3, d)
        T = x.shape[1]
        pos = self.triple_pos(torch.arange(T, device=x.device).clamp_max(self.cfg.max_triples - 1))
        x = x + self.role.weight.view(1, 1, 3, -1) + pos.view(1, T, 1, -1)
        B, _, _, d = x.shape
        real = b["tok_mask"].repeat_interleave(3, dim=1)
        is_state = (b["tok_mask"] & torch.isin(b["scope"], self.state_scopes)).repeat_interleave(3, dim=1)
        return x.reshape(B, T * 3, d), real, is_state


class DecisionMLP(nn.Module):
    def __init__(self, vocab: DecisionVocab, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        d, h = cfg.d_model, cfg.mlp_hidden
        self.embed = TripleEmbedding(vocab, d)
        # φ: one triple's parts (3d) to a vector (d), applied to each triple alone.
        self.phi = nn.Sequential(nn.Linear(3 * d, h), nn.GELU(), nn.Linear(h, d))
        # ρ: dropout after each hidden layer; at play time (eval) it is the
        # identity, so the exported layout (mlp.0, .1, .4, .7) is what the JS reads.
        p = cfg.dropout
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(h, d))

    def forward(self, b: dict) -> torch.Tensor:
        x = (self.phi(self.embed.triple_parts(b)) * b["tok_mask"].unsqueeze(-1)).sum(dim=1)
        z = self.mlp(x)
        logits = torch.einsum("bd,bkd->bk", z, self.embed.options(b))
        return logits.masked_fill(~b["opt_mask"], float("-inf"))


class DecisionFM(nn.Module):
    def __init__(self, vocab: DecisionVocab, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = TripleEmbedding(vocab, d)
        self.pair_norm = nn.LayerNorm(d)
        self.sum_norm = nn.LayerNorm(d)
        if cfg.fm_order >= 3:
            self.third = nn.Linear(d, d, bias=False)
            self.third_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, b: dict) -> torch.Tensor:
        mask = b["tok_mask"].unsqueeze(-1)
        e = self.embed.triples(b) * mask
        s = e.sum(dim=1)
        pairs = 0.5 * (s * s - (e * e).sum(dim=1))
        h = self.pair_norm(pairs) + self.sum_norm(s)
        if self.cfg.fm_order >= 3:
            f = self.third(e) * mask
            p1, p2, p3 = f.sum(dim=1), (f * f).sum(dim=1), (f * f * f).sum(dim=1)
            h = h + self.third_norm((p1 ** 3 - 3 * p1 * p2 + 2 * p3) / 6)
        h = self.drop(h)
        logits = torch.einsum("bd,bkd->bk", h, self.embed.options(b))
        return logits.masked_fill(~b["opt_mask"], float("-inf"))


def build_student(vocab: DecisionVocab, cfg: StudentConfig) -> nn.Module:
    if cfg.arch == "set":
        return DecisionSetModel(vocab, cfg)
    if cfg.arch == "twm":
        return DecisionTWMModel(vocab, cfg)
    if cfg.arch == "mlp":
        return DecisionMLP(vocab, cfg)
    if cfg.arch == "fm":
        return DecisionFM(vocab, cfg)
    raise ValueError(f"unknown arch {cfg.arch}")


def kl_teacher_student(logits: torch.Tensor, target: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
    """Per-row KL(teacher || student) in nats over the offered options."""
    logq = F.log_softmax(logits, dim=-1).masked_fill(~opt_mask, 0.0)
    p = target * opt_mask
    plogp = torch.where(p > 0, p * torch.log(p.clamp_min(1e-12)), torch.zeros_like(p))
    return (plogp - p * logq).sum(dim=-1)
