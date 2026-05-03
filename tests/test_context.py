"""
Tests for src/llm/context.py (Phase 1.9)

All tests are pure-Python — no Ollama dependency.
"""

from pathlib import Path

import pytest
import yaml

from src.llm.context import (
    CHARS_PER_TOKEN,
    AssembledContext,
    ContextBudget,
    count_tokens,
)
from src.memory.soul import SoulFile


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

def test_count_tokens_empty_string() -> None:
    assert count_tokens("") == 0


def test_count_tokens_known_length() -> None:
    # 40 chars → 10 tokens at CHARS_PER_TOKEN=4
    assert count_tokens("a" * 40) == 40 // CHARS_PER_TOKEN


def test_count_tokens_uses_chars_per_token_constant() -> None:
    text = "x" * 100
    assert count_tokens(text) == 100 // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# ContextBudget — budget helper properties
# ---------------------------------------------------------------------------

def test_budget_allocations_sum_to_usable_tokens() -> None:
    b = ContextBudget(total_tokens=4096, response_reserve=512)
    usable_chars = (4096 - 512) * CHARS_PER_TOKEN
    total_alloc = (
        b.soul_budget_chars()
        + b.history_budget_chars()
        + b.rag_vision_budget_chars()
        + b.misc_budget_chars()
    )
    # Allow rounding error of up to 1 char per tier (4 tiers × CHARS_PER_TOKEN possible floor error)
    assert abs(total_alloc - usable_chars) <= 4 * CHARS_PER_TOKEN


def test_response_reserve_is_always_protected() -> None:
    b = ContextBudget(total_tokens=512, response_reserve=512)
    # Usable = 0 → all budgets should be 0
    assert b.soul_budget_chars() == 0
    assert b.history_budget_chars() == 0


# ---------------------------------------------------------------------------
# ContextBudget.assemble — all sections fit
# ---------------------------------------------------------------------------

def test_assemble_no_trimming_when_all_fit() -> None:
    b = ContextBudget(total_tokens=4096, response_reserve=512)
    soul = "S" * 10
    history = [{"role": "user", "content": "hello"}]
    rag = "R" * 10
    misc = "M" * 10

    ctx = b.assemble(soul_text=soul, history=history, rag_text=rag, misc_text=misc)

    assert ctx.was_trimmed is False
    assert ctx.soul == soul
    assert ctx.rag == rag
    assert ctx.misc == misc


# ---------------------------------------------------------------------------
# ContextBudget.assemble — misc over budget
# ---------------------------------------------------------------------------

def test_assemble_truncates_misc_when_over_budget() -> None:
    b = ContextBudget(total_tokens=100, response_reserve=0, misc_pct=0.10)
    # misc budget = 100 * 0.10 * 4 = 40 chars
    misc_budget = b.misc_budget_chars()
    over_budget_misc = "M" * (misc_budget + 100)

    ctx = b.assemble(soul_text="", history=[], misc_text=over_budget_misc)

    assert len(ctx.misc) <= misc_budget
    assert "misc" in ctx.trimmed


# ---------------------------------------------------------------------------
# ContextBudget.assemble — vision trimmed before RAG
# ---------------------------------------------------------------------------

def test_assemble_drops_vision_before_rag() -> None:
    b = ContextBudget(total_tokens=100, response_reserve=0, rag_vision_pct=0.10)
    rv_budget = b.rag_vision_budget_chars()
    # Fill the budget with rag; add vision that pushes it over
    rag = "R" * (rv_budget - 5)
    vision = "V. Extra vision content that definitely overflows the budget."

    ctx = b.assemble(soul_text="", history=[], rag_text=rag, vision_text=vision)

    # Vision should have been cut; rag preserved
    assert "vision" in ctx.trimmed
    assert ctx.rag == rag


# ---------------------------------------------------------------------------
# SoulFile.to_prompt_section — budget trimming levels
# ---------------------------------------------------------------------------

def test_to_prompt_section_drops_facts_first(tmp_path: Path) -> None:
    """Level 1 trim: capabilities/hardware dropped before other identity fields."""
    data = {
        "identity": {
            "name": "Orion",
            "persona": "warm robot",
            "communication_style": "concise",
            "capabilities": ["should be dropped"],
        },
    }
    soul = SoulFile(path=tmp_path / "soul.yaml")
    soul.save(data)

    full = soul.to_prompt_section()
    # Budget just enough to force Level-1 trim (drop capabilities)
    trimmed_text = soul.to_prompt_section(budget_chars=len(full) - 5)

    assert "should be dropped" not in trimmed_text


def test_to_prompt_section_returns_full_when_no_budget(tmp_path: Path) -> None:
    data = {"identity": {"name": "Orion"}, "user": {"name": "Daniel"}}
    soul = SoulFile(path=tmp_path / "soul.yaml")
    soul.save(data)

    assert soul.to_prompt_section() == soul.to_prompt_section(budget_chars=999_999)
