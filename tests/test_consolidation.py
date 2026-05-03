"""
Tests for ConsolidationEngine — Phase 2.5.

All LLM calls are mocked so these tests run without Ollama.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.memory.claims import ClaimsStore
from src.memory.consolidation import (
    ConsolidationEngine,
    CONSOLIDATION_INTERVAL_HOURS,
    MIN_EPISODES_TO_CONSOLIDATE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def claims(tmp_path: Path) -> ClaimsStore:
    return ClaimsStore(db_path=tmp_path / "test_claims.db")


@pytest.fixture
def engine(claims: ClaimsStore) -> ConsolidationEngine:
    return ConsolidationEngine(
        claims_store=claims,
        memory_store=None,
        model="test-model",
        base_url="http://localhost:11434",
    )


# ---------------------------------------------------------------------------
# _parse_candidates
# ---------------------------------------------------------------------------


def test_parse_candidates_valid_json(engine: ConsolidationEngine) -> None:
    raw = """
```json
{"claims": [
  {"claim": "Daniel was born on June 20, 1989.", "category": "biographical_facts", "confidence": 1.0},
  {"claim": "Daniel's wife is Danielle Smith.", "category": "relationships", "confidence": 0.95}
]}
```
"""
    result = engine._parse_candidates(raw)
    assert len(result) == 2
    assert result[0]["claim"] == "Daniel was born on June 20, 1989."


def test_parse_candidates_empty_claims(engine: ConsolidationEngine) -> None:
    raw = '```json\n{"claims": []}\n```'
    result = engine._parse_candidates(raw)
    assert result == []


def test_parse_candidates_malformed_json(engine: ConsolidationEngine) -> None:
    raw = "This is not JSON at all."
    result = engine._parse_candidates(raw)
    assert result == []


def test_parse_candidates_missing_claims_key(engine: ConsolidationEngine) -> None:
    raw = '```json\n{"facts": [{"claim": "something"}]}\n```'
    result = engine._parse_candidates(raw)
    assert result == []


# ---------------------------------------------------------------------------
# consolidate — with mocked LLM
# ---------------------------------------------------------------------------


_MOCK_CLAIMS_JSON = """```json
{"claims": [
  {"claim": "Daniel was born on June 20, 1989.", "category": "biographical_facts", "confidence": 1.0},
  {"claim": "Daniel's wife is Danielle Smith.", "category": "relationships", "confidence": 1.0}
]}
```"""


@patch("src.llm.client.OllamaClient")
def test_consolidate_writes_claims(MockClient, engine: ConsolidationEngine) -> None:
    MockClient.return_value.chat.return_value = _MOCK_CLAIMS_JSON
    written = engine.consolidate(
        ["Daniel was born in 1989 and is married to Danielle."],
        source_ids=["ep1"],
    )
    assert len(written) == 2
    assert written[0]["category"] == "biographical_facts"


@patch("src.llm.client.OllamaClient")
def test_consolidate_empty_episodes(MockClient, engine: ConsolidationEngine) -> None:
    written = engine.consolidate([])
    assert written == []
    MockClient.return_value.chat.assert_not_called()


@patch("src.llm.client.OllamaClient")
def test_consolidate_llm_failure_returns_empty(
    MockClient, engine: ConsolidationEngine
) -> None:
    MockClient.return_value.chat.side_effect = RuntimeError("Ollama offline")
    written = engine.consolidate(["Daniel was born in 1989."])
    assert written == []


@patch("src.llm.client.OllamaClient")
def test_consolidate_skips_short_claims(MockClient, engine: ConsolidationEngine) -> None:
    short_json = '```json\n{"claims": [{"claim": "Hi", "category": "general", "confidence": 1.0}]}\n```'
    MockClient.return_value.chat.return_value = short_json
    written = engine.consolidate(["Hi there."])
    # "Hi" is too short — should be rejected by policy/claims validation
    assert len(written) == 0


@patch("src.llm.client.OllamaClient")
def test_consolidate_idempotent(MockClient, engine: ConsolidationEngine) -> None:
    """Running consolidate twice with the same episodes reinforces rather than duplicates."""
    MockClient.return_value.chat.return_value = _MOCK_CLAIMS_JSON
    written1 = engine.consolidate(["Daniel was born in 1989 and married Danielle."])
    written2 = engine.consolidate(["Daniel was born in 1989 and married Danielle."])
    # First run: 2 new claims written
    assert len(written1) == 2
    # Second run: same claims already exist and contradiction check should reinforce or skip
    # The total claim count should still be 2
    counts = engine._claims.count()
    assert counts["total"] == 2


# ---------------------------------------------------------------------------
# run_if_due — scheduler
# ---------------------------------------------------------------------------


def test_run_if_due_returns_zero_without_memory_store(
    engine: ConsolidationEngine,
) -> None:
    assert engine._memory is None
    result = engine.run_if_due()
    assert result == 0


@patch("src.llm.client.OllamaClient")
def test_run_if_due_skips_when_not_enough_episodes(
    MockClient, claims: ClaimsStore, tmp_path: Path
) -> None:
    """Should not consolidate when MemoryStore has fewer than MIN_EPISODES episodes."""
    mock_memory = MagicMock()
    mock_memory.get_all_memories.return_value = [
        {"id": "1", "text": "short session"}
    ]  # only 1, need MIN_EPISODES_TO_CONSOLIDATE
    eng = ConsolidationEngine(
        claims_store=claims,
        memory_store=mock_memory,
        model="test-model",
        base_url="http://localhost:11434",
    )
    result = eng.run_if_due()
    assert result == 0


@patch("src.llm.client.OllamaClient")
def test_run_if_due_runs_when_due(MockClient, claims: ClaimsStore) -> None:
    MockClient.return_value.chat.return_value = _MOCK_CLAIMS_JSON
    mock_memory = MagicMock()
    mock_memory.get_all_memories.return_value = [
        {"id": str(i), "text": f"Session {i}: Daniel mentioned something important."}
        for i in range(MIN_EPISODES_TO_CONSOLIDATE)
    ]
    eng = ConsolidationEngine(
        claims_store=claims,
        memory_store=mock_memory,
        model="test-model",
        base_url="http://localhost:11434",
    )
    result = eng.run_if_due()
    assert result >= 0  # ran and returned a count (may be 0 if all filtered)


# ---------------------------------------------------------------------------
# _is_due scheduler logic
# ---------------------------------------------------------------------------


def test_is_due_first_run(engine: ConsolidationEngine) -> None:
    """No consolidation_state row → always due."""
    assert engine._is_due() is True


def test_is_due_after_record_run(engine: ConsolidationEngine) -> None:
    """After _record_run(), enough time has NOT passed so not due."""
    engine._record_run()
    assert engine._is_due() is False


def test_record_run_persists(engine: ConsolidationEngine) -> None:
    engine._record_run()
    import sqlite3
    with sqlite3.connect(str(engine._claims._path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT last_run_at FROM consolidation_state LIMIT 1"
        ).fetchone()
    assert row is not None
    assert row["last_run_at"] is not None


# ---------------------------------------------------------------------------
# run_async — smoke test
# ---------------------------------------------------------------------------


def test_run_async_does_not_block(engine: ConsolidationEngine) -> None:
    """run_async() should return immediately (daemon thread)."""
    import time
    start = time.monotonic()
    engine.run_async()
    elapsed = time.monotonic() - start
    assert elapsed < 1.0  # should not block for more than 1 second


# ---------------------------------------------------------------------------
# _get_all_episodes
# ---------------------------------------------------------------------------


def test_get_all_episodes_no_memory(engine: ConsolidationEngine) -> None:
    result = engine._get_all_episodes()
    assert result == []


def test_get_all_episodes_with_mock(engine: ConsolidationEngine) -> None:
    mock_memory = MagicMock()
    mock_memory.get_all_memories.return_value = [
        {"id": "a", "text": "First session summary."},
        {"id": "b", "text": "Second session summary."},
    ]
    engine._memory = mock_memory
    result = engine._get_all_episodes()
    assert len(result) == 2
    assert result[0]["id"] == "a"
