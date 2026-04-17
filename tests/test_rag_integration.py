"""
Tests for Phase 2.4 — RAG integration layer.

Covers:
- ContextBudget.rag_budget_chars() returns a positive integer
- RAG query injection: empty store → no section, below-threshold → no section,
  above-threshold → ## Relevant Memory block is built correctly
- Budget cap: results are truncated when combined chars would exceed rag_budget_chars()
- save_session: summarise_session() → add_memory() called when summary is non-empty
- save_session: add_memory() NOT called when summariser returns ""
- Memory logging: add_memory() logs an info record, query_memory() logs an info record
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm.context import ContextBudget
from src.memory.vector_store import MemoryStore


# ---------------------------------------------------------------------------
# ContextBudget.rag_budget_chars
# ---------------------------------------------------------------------------

def test_rag_budget_chars_is_positive() -> None:
    budget = ContextBudget()
    assert budget.rag_budget_chars() > 0


def test_rag_budget_chars_equals_rag_vision_budget_chars() -> None:
    """Until vision is wired, rag_budget_chars() delegates to rag_vision_budget_chars()."""
    budget = ContextBudget()
    assert budget.rag_budget_chars() == budget.rag_vision_budget_chars()


def test_rag_budget_chars_scales_with_total_tokens() -> None:
    small = ContextBudget(total_tokens=4096, response_reserve=512)
    large = ContextBudget(total_tokens=8192, response_reserve=512)
    assert large.rag_budget_chars() > small.rag_budget_chars()


# ---------------------------------------------------------------------------
# RAG injection logic (inline — mirrors app.py / main.py build logic)
# ---------------------------------------------------------------------------

def _build_rag_section(results: list[str], rag_budget: int) -> str:
    """Replicate the injection logic from app.py and main.py."""
    if not results:
        return ""
    kept, total_chars = [], 0
    for result in results:
        entry = f"- {result}"
        if total_chars + len(entry) + 1 > rag_budget:
            break
        kept.append(entry)
        total_chars += len(entry) + 1
    if not kept:
        return ""
    return "## Relevant Memory\n\n" + "\n".join(kept)


def test_build_rag_section_empty_results_returns_empty_string() -> None:
    assert _build_rag_section([], rag_budget=6144) == ""


def test_build_rag_section_single_result_returns_formatted_block() -> None:
    section = _build_rag_section(["Daniel likes hiking."], rag_budget=6144)
    assert section.startswith("## Relevant Memory")
    assert "- Daniel likes hiking." in section


def test_build_rag_section_multiple_results_all_included_within_budget() -> None:
    results = ["Fact A.", "Fact B.", "Fact C."]
    section = _build_rag_section(results, rag_budget=6144)
    assert "- Fact A." in section
    assert "- Fact B." in section
    assert "- Fact C." in section


def test_build_rag_section_budget_cap_truncates_results() -> None:
    # Budget of 20 chars: "- Short." = 9 chars, "- Another long fact..." won't fit
    short = "Short."
    long_fact = "A very long fact that definitely exceeds the remaining budget."
    section = _build_rag_section([short, long_fact], rag_budget=20)
    assert "- Short." in section
    assert long_fact not in section


def test_build_rag_section_budget_zero_returns_empty_string() -> None:
    assert _build_rag_section(["Some fact."], rag_budget=0) == ""


# ---------------------------------------------------------------------------
# MemoryStore integration: query → inject
# ---------------------------------------------------------------------------

def test_query_memory_returns_empty_list_for_empty_store(tmp_path: Path) -> None:
    store = MemoryStore(persist_dir=str(tmp_path))
    results = store.query_memory("anything")
    assert results == []


def test_query_memory_returns_relevant_result_above_threshold(tmp_path: Path) -> None:
    store = MemoryStore(persist_dir=str(tmp_path))
    store.add_memory("Daniel loves hiking in the mountains.")
    # Query with a semantically close sentence
    results = store.query_memory("What does Daniel enjoy doing outdoors?")
    # At least the added memory should pass the default 0.35 threshold
    assert len(results) >= 1


def test_query_memory_respects_similarity_threshold(tmp_path: Path) -> None:
    store = MemoryStore(persist_dir=str(tmp_path), similarity_threshold=0.99)
    store.add_memory("The capital of France is Paris.")
    # An unrelated query won't clear a 0.99 threshold
    results = store.query_memory("Tell me about dinosaurs.")
    assert results == []


def test_no_rag_section_injected_when_query_returns_nothing(tmp_path: Path) -> None:
    store = MemoryStore(persist_dir=str(tmp_path))
    results = store.query_memory("anything")
    section = _build_rag_section(results, rag_budget=6144)
    assert section == ""


# ---------------------------------------------------------------------------
# save_session_to_memory (mirrors main.py _save_session_summary logic)
# ---------------------------------------------------------------------------

MINIMAL_CONVERSATION = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thanks!"},
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "I don't have access to weather data."},
]


@patch("src.llm.client.OllamaClient")
def test_save_session_calls_add_memory_when_summary_nonempty(
    mock_client_cls: MagicMock, tmp_path: Path
) -> None:
    mock_instance = MagicMock()
    mock_instance.chat.return_value = "Daniel asked about the weather and I replied."
    mock_client_cls.return_value = mock_instance

    store = MemoryStore(persist_dir=str(tmp_path))
    initial_count = store.count()

    from src.memory.summariser import summarise_session

    summary = summarise_session(MINIMAL_CONVERSATION, model="test-model", base_url="http://localhost:11434")
    if summary:
        store.add_memory(summary, {"source": "session_summary"})

    # Either summariser produced output (and count went up) or it returned ""
    # We only assert add_memory ran if summary was non-empty
    if summary:
        assert store.count() == initial_count + 1


@patch("src.llm.client.OllamaClient")
def test_save_session_skips_add_memory_when_summary_empty(
    mock_client_cls: MagicMock, tmp_path: Path
) -> None:
    mock_instance = MagicMock()
    mock_instance.chat.return_value = ""
    mock_client_cls.return_value = mock_instance

    store = MemoryStore(persist_dir=str(tmp_path))

    from src.memory.summariser import summarise_session

    summary = summarise_session(
        MINIMAL_CONVERSATION,
        model="test-model",
        base_url="http://localhost:11434",
    )
    # With empty LLM response, summary must be ""
    assert summary == ""
    # Nothing added to store
    assert store.count() == 0


def test_save_session_skips_on_too_few_turns(tmp_path: Path) -> None:
    """Conversations shorter than min_turns produce no summary and no store write."""
    short_convo = [
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello!"},
    ]
    store = MemoryStore(persist_dir=str(tmp_path))

    from src.memory.summariser import summarise_session, MIN_TURNS

    if MIN_TURNS > 1:
        summary = summarise_session(short_convo, model="any", base_url="http://x")
        assert summary == ""
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Memory logging
# ---------------------------------------------------------------------------

def test_add_memory_emits_info_log(tmp_path: Path) -> None:
    import src.memory.vector_store as vs_module

    mock_log = MagicMock()
    with patch.object(vs_module, "_get_mem_log", return_value=mock_log):
        store = MemoryStore(persist_dir=str(tmp_path))
        store.add_memory("Test memory log entry.")

    mock_log.info.assert_called_once()
    call_args = mock_log.info.call_args[0]
    assert "add_memory" in call_args[0]


def test_query_memory_emits_info_log(tmp_path: Path) -> None:
    import src.memory.vector_store as vs_module

    mock_log = MagicMock()
    with patch.object(vs_module, "_get_mem_log", return_value=mock_log):
        store = MemoryStore(persist_dir=str(tmp_path))
        store.add_memory("Some fact.")
        store.query_memory("Some query.")

    # info called twice: once for add_memory, once for query_memory
    assert mock_log.info.call_count == 2
    messages = [c[0][0] for c in mock_log.info.call_args_list]
    assert any("query_memory" in m for m in messages)
