"""Bridge between an LLM and TWM for structured reasoning.

TWM as reasoning middleware: the LLM handles messy text↔structure conversion,
TWM handles the structured state transition prediction.

Usage:
    from twm.llm_bridge import TWMBridge

    bridge = TWMBridge(checkpoint_dir="results/08_context_dependent")
    result = bridge.reason("Alice has a full glass and she's thirsty")
    print(result)
    # "Alice drinks the water. The glass becomes empty and Alice is satisfied."

    # Or step by step:
    triples = bridge.decompose("Alice has a full glass and she's thirsty")
    # [["alice", "state", "thirsty"], ["glass", "state", "full"]]
    next_state = bridge.predict(triples)
    # [["alice", "state", "satisfied"], ["glass", "state", "empty"]]
    explanation = bridge.interpret(triples, next_state)
"""

import json
import os
from pathlib import Path

from .serve import WorldModel


DECOMPOSE_SYSTEM = """You convert natural language situation descriptions into structured triples.
Each triple is [entity, attribute, value] where each element is a single lowercase word or underscore-separated phrase.

Rules:
- Use 1-6 triples to capture the key entities and their states
- Entity names: lowercase, use underscores (alice, glass, room_door)
- Attributes: state, location, action, property, relation, intent
- Values: single descriptive tokens (full, empty, thirsty, satisfied, open, closed)
- Focus on STATE, not narrative. What are the current conditions?

Output ONLY a JSON list of triples, nothing else."""

DECOMPOSE_USER = """Convert this situation to triples:
"{situation}"

JSON list of [entity, attribute, value] triples:"""

INTERPRET_SYSTEM = """You explain state transitions in natural language.
Given a before-state and after-state (as structured triples), describe what happened and why.
Be concise — 1-2 sentences. Focus on what changed and the causal chain."""

INTERPRET_USER = """Before state:
{before}

After state:
{after}

What happened?"""


class TWMBridge:
    """Bridge between an LLM and TWM for structured reasoning."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        api_key: str | None = None,
        llm_model: str = "claude-sonnet-4-20250514",
        device: str | None = None,
    ):
        self.wm = WorldModel(checkpoint_dir, device=device)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.llm_model = llm_model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required for LLM bridge. "
                    "Install with: uv pip install anthropic"
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _llm_call(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.llm_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()

    def decompose(self, situation: str) -> list[list[str]]:
        """LLM call: convert natural language to TWM triples."""
        text = self._llm_call(DECOMPOSE_SYSTEM, DECOMPOSE_USER.format(situation=situation))

        # Parse JSON from response
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse triples from LLM response: {text}")

        triples = json.loads(match.group())

        # Validate and normalize
        result = []
        for triple in triples:
            if isinstance(triple, list) and len(triple) == 3:
                result.append([str(t).strip().lower().replace(" ", "_") for t in triple])
        return result

    def predict(self, triples: list[list[str]]) -> list[list[str]]:
        """TWM call: predict next state."""
        return self.wm.advance(triples)

    def predict_n(self, triples: list[list[str]], n_steps: int = 1) -> list[list[list[str]]]:
        """TWM call: predict n steps forward."""
        return self.wm.advance_n(triples, n_steps)

    def interpret(self, before: list[list[str]], after: list[list[str]]) -> str:
        """LLM call: explain what changed and why."""
        before_str = json.dumps(before, indent=2)
        after_str = json.dumps(after, indent=2)
        return self._llm_call(
            INTERPRET_SYSTEM,
            INTERPRET_USER.format(before=before_str, after=after_str),
        )

    def reason(self, situation: str, n_steps: int = 1) -> dict:
        """Full pipeline: decompose -> predict -> interpret.

        Returns dict with all intermediate results for transparency.
        """
        triples = self.decompose(situation)

        if n_steps == 1:
            predicted = self.predict(triples)
            explanation = self.interpret(triples, predicted)
            return {
                "input_situation": situation,
                "decomposed_triples": triples,
                "predicted_state": predicted,
                "explanation": explanation,
            }
        else:
            trajectory = self.predict_n(triples, n_steps)
            explanations = []
            prev = triples
            for step_state in trajectory:
                exp = self.interpret(prev, step_state)
                explanations.append(exp)
                prev = step_state

            return {
                "input_situation": situation,
                "decomposed_triples": triples,
                "trajectory": trajectory,
                "explanations": explanations,
            }

    def reason_no_llm(self, triples: list[list[str]], n_steps: int = 1) -> dict:
        """Predict without LLM — for when triples are already structured.

        Useful for benchmarking TWM's prediction without LLM overhead.
        """
        if n_steps == 1:
            predicted = self.predict(triples)
            return {
                "input_triples": triples,
                "predicted_state": predicted,
            }
        else:
            trajectory = self.predict_n(triples, n_steps)
            return {
                "input_triples": triples,
                "trajectory": trajectory,
            }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TWM-LLM Bridge")
    parser.add_argument("--checkpoint", required=True, help="TWM checkpoint directory")
    parser.add_argument("--situation", type=str, help="Natural language situation to reason about")
    parser.add_argument("--triples", type=str, help="JSON triples (skip decompose step)")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="LLM model for decompose/interpret")

    args = parser.parse_args()
    bridge = TWMBridge(args.checkpoint, llm_model=args.model, device=args.device)

    if args.triples:
        triples = json.loads(args.triples)
        result = bridge.reason_no_llm(triples, n_steps=args.steps)
    elif args.situation:
        result = bridge.reason(args.situation, n_steps=args.steps)
    else:
        parser.error("Provide either --situation or --triples")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
