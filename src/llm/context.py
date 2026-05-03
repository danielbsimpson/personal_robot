"""
Context budget management for the personal robot LLM.

Measures every section of the system prompt before it is sent, ensures the
model always has headroom to reply, and degrades gracefully — dropping the
least-important content first — when the budget is tight.

Token counting is approximated as len(text) // CHARS_PER_TOKEN (1 token ≈ 4
chars). A more accurate counter (e.g. tiktoken) can be swapped in later by
replacing ``count_tokens`` without changing any call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN: int = 4


def count_tokens(text: str) -> int:
    """Approximate token count for *text* using a fixed chars-per-token ratio.

    1 token ≈ 4 characters — fast, zero-dependency, easy to swap for tiktoken.
    """
    return len(text) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# AssembledContext result type
# ---------------------------------------------------------------------------

@dataclass
class AssembledContext:
    """Output of ``ContextBudget.assemble()`` — budget-checked sections."""

    soul: str
    history: list[dict]
    rag: str
    vision: str
    misc: str

    trimmed: set[str] = field(default_factory=set)

    @property
    def was_trimmed(self) -> bool:
        """True if any section was cut to fit the budget."""
        return bool(self.trimmed)

    @property
    def total_chars(self) -> int:
        """Total character count across all text sections (history excluded)."""
        return sum(len(s) for s in (self.soul, self.rag, self.vision, self.misc))


# ---------------------------------------------------------------------------
# ContextBudget
# ---------------------------------------------------------------------------

@dataclass
class ContextBudget:
    """Percentage-based context assembly that guarantees response headroom.

    Tier allocations (applied to ``total_tokens - response_reserve``):

    +--------------+-------+--------------------------------------------------+
    | Tier         | Share | Contents                                         |
    +==============+=======+==================================================+
    | soul/system  |  20 % | BASE_SYSTEM_PROMPT + soul file ## About Me       |
    | history      |  50 % | Rolling message list, oldest first               |
    | rag + vision |  20 % | ## Relevant Memory + ## Current Environment      |
    | misc         |  10 % | ## Current Time and any future injected sections |
    +--------------+-------+--------------------------------------------------+

    The response reserve (default 512 tokens) is always protected — it is
    subtracted before any percentage is calculated.
    """

    total_tokens: int = 8192
    response_reserve: int = 512

    soul_pct: float = 0.35
    history_pct: float = 0.35
    rag_vision_pct: float = 0.20
    misc_pct: float = 0.10

    def _usable_tokens(self) -> int:
        return max(0, self.total_tokens - self.response_reserve)

    def soul_budget_chars(self) -> int:
        """Character budget for the soul + base system prompt section."""
        return int(self._usable_tokens() * self.soul_pct) * CHARS_PER_TOKEN

    def history_budget_chars(self) -> int:
        """Character budget for the rolling conversation history."""
        return int(self._usable_tokens() * self.history_pct) * CHARS_PER_TOKEN

    def rag_vision_budget_chars(self) -> int:
        """Character budget shared by RAG memories and vision context."""
        return int(self._usable_tokens() * self.rag_vision_pct) * CHARS_PER_TOKEN

    def rag_budget_chars(self) -> int:
        """Character budget for RAG (Relevant Memory) text only.

        Uses the full rag_vision allocation until Phase 5 adds vision — at
        that point the two will share the budget 50/50.
        """
        return self.rag_vision_budget_chars()

    def claims_budget_chars(self) -> int:
        """Character budget for trust-calibrated claims (## Long-Term Knowledge).

        Phase 2.5: 40 % of the rag+vision budget is reserved for claims so
        the most reliable long-term facts get priority context space.
        """
        return int(self.rag_vision_budget_chars() * 0.40)

    def facts_budget_chars(self) -> int:
        """Character budget for keyword-triggered structured facts (## Relevant Facts).

        Phase 2.5: 30 % of the rag+vision budget.
        """
        return int(self.rag_vision_budget_chars() * 0.30)

    def episodes_budget_chars(self) -> int:
        """Character budget for session-summary episode memories (## Relevant Memory).

        Phase 2.5: 30 % of the rag+vision budget.
        """
        return int(self.rag_vision_budget_chars() * 0.30)

    def misc_budget_chars(self) -> int:
        """Character budget for time/misc sections."""
        return int(self._usable_tokens() * self.misc_pct) * CHARS_PER_TOKEN

    def assemble(
        self,
        soul_text: str,
        history: list[dict],
        rag_text: str = "",
        vision_text: str = "",
        misc_text: str = "",
    ) -> AssembledContext:
        """Measure each section against its budget and trim as needed.

        Trimming order (least important first):
          1. RAG text — reduced to first sentence, then dropped
          2. Vision text — truncated to first sentence, then dropped
          3. Misc text — truncated, then dropped
          4. Soul text — delegated to ``SoulFile.to_prompt_section(budget_chars)``
             (caller must pass pre-trimmed soul text or handle it separately)
          5. History — oldest pairs dropped (caller passes already-trimmed list,
             or downstream ``trim_history(budget_chars=...)`` handles it)

        This method trims rag/vision/misc in-place and records what was cut.
        Soul and history trimming are handled at their own call sites because
        they require richer logic (progressive YAML dropping, pair trimming).

        Returns an ``AssembledContext`` with trimmed flags set.
        """
        trimmed: set[str] = set()

        # --- misc ---
        misc_budget = self.misc_budget_chars()
        if len(misc_text) > misc_budget:
            misc_text = misc_text[:misc_budget]
            trimmed.add("misc")

        # --- rag + vision (shared budget) ---
        rv_budget = self.rag_vision_budget_chars()
        combined_rv = len(rag_text) + len(vision_text)
        if combined_rv > rv_budget:
            # Trim vision first: first sentence then drop
            if vision_text:
                first_sentence = (vision_text.split(".")[0] + ".").strip()
                if len(rag_text) + len(first_sentence) <= rv_budget:
                    vision_text = first_sentence
                else:
                    vision_text = ""
                trimmed.add("vision")
            # If still over, trim rag to first sentence then drop
            if len(rag_text) + len(vision_text) > rv_budget:
                if rag_text:
                    first_sentence = (rag_text.split(".")[0] + ".").strip()
                    if len(first_sentence) + len(vision_text) <= rv_budget:
                        rag_text = first_sentence
                    else:
                        rag_text = ""
                    trimmed.add("rag")

        # --- soul: flag if over budget; caller (SoulFile.to_prompt_section)
        #     does the actual trimming so we just record the outcome ---
        soul_budget = self.soul_budget_chars()
        if len(soul_text) > soul_budget:
            trimmed.add("soul")

        # --- history: flag if over budget; trim_history does the work ---
        history_budget = self.history_budget_chars()
        history_chars = sum(len(m["content"]) for m in history)
        if history_chars > history_budget:
            trimmed.add("history")

        return AssembledContext(
            soul=soul_text,
            history=history,
            rag=rag_text,
            vision=vision_text,
            misc=misc_text,
            trimmed=trimmed,
        )
