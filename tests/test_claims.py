"""
Tests for ClaimsStore — Phase 2.5.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

import pytest

from src.memory.claims import (
    ClaimsStore,
    _claim_id,
    _trust_score,
    DECAY_DAYS,
    VECTOR_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> ClaimsStore:
    """Return a fresh ClaimsStore backed by a temporary SQLite database."""
    return ClaimsStore(db_path=tmp_path / "test_claims.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CLAIM = "Daniel's wife is Danielle Smith."
SHORT_TEXT = "Hi"  # too short to store


# ---------------------------------------------------------------------------
# Schema and bootstrap
# ---------------------------------------------------------------------------


def test_schema_creates_tables(store: ClaimsStore) -> None:
    with store._connect() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "claims" in tables
    assert "claim_events" in tables
    assert "schema_version" in tables


def test_schema_version_inserted(store: ClaimsStore) -> None:
    with store._connect() as conn:
        ver = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    assert ver is not None
    assert ver[0] >= 1


# ---------------------------------------------------------------------------
# add_claim — basic CRUD
# ---------------------------------------------------------------------------


def test_add_claim_returns_id(store: ClaimsStore) -> None:
    cid = store.add_claim(SAMPLE_CLAIM)
    assert isinstance(cid, str)
    assert len(cid) == 32


def test_add_claim_stores_text(store: ClaimsStore) -> None:
    store.add_claim(SAMPLE_CLAIM, category="relationships", confidence=0.9)
    with store._connect() as conn:
        row = conn.execute("SELECT * FROM claims LIMIT 1").fetchone()
    assert row["text"] == SAMPLE_CLAIM
    assert row["category"] == "relationships"
    assert abs(row["confidence"] - 0.9) < 1e-6


def test_add_claim_creates_event(store: ClaimsStore) -> None:
    cid = store.add_claim(SAMPLE_CLAIM)
    with store._connect() as conn:
        events = conn.execute(
            "SELECT * FROM claim_events WHERE claim_id = ?", (cid,)
        ).fetchall()
    assert len(events) == 1
    assert events[0]["event_type"] == "created"


def test_add_claim_short_text_raises(store: ClaimsStore) -> None:
    with pytest.raises(ValueError):
        store.add_claim(SHORT_TEXT)


def test_add_claim_empty_raises(store: ClaimsStore) -> None:
    with pytest.raises(ValueError):
        store.add_claim("")


# ---------------------------------------------------------------------------
# Idempotency — duplicate adds reinforce the existing claim
# ---------------------------------------------------------------------------


def test_add_claim_idempotent_reinforces(store: ClaimsStore) -> None:
    cid1 = store.add_claim(SAMPLE_CLAIM)
    cid2 = store.add_claim(SAMPLE_CLAIM)
    assert cid1 == cid2  # same normalised text → same ID
    with store._connect() as conn:
        row = conn.execute("SELECT reinforcement_count FROM claims WHERE id = ?", (cid1,)).fetchone()
    assert row["reinforcement_count"] == 1  # incremented once


def test_add_claim_normalisation(store: ClaimsStore) -> None:
    """Leading/trailing whitespace and different capitalisation → same ID."""
    cid1 = store.add_claim("Daniel's wife is Danielle Smith.")
    cid2 = store.add_claim("  daniel's wife is danielle smith.  ")
    assert cid1 == cid2


def test_reinforce_merges_source_ids(store: ClaimsStore) -> None:
    cid = store.add_claim(SAMPLE_CLAIM, source_ids=["ep1"])
    store.add_claim(SAMPLE_CLAIM, source_ids=["ep2"])
    with store._connect() as conn:
        row = conn.execute(
            "SELECT source_episode_ids FROM claims WHERE id = ?", (cid,)
        ).fetchone()
    merged = json.loads(row["source_episode_ids"])
    assert "ep1" in merged
    assert "ep2" in merged


# ---------------------------------------------------------------------------
# reinforce_claim
# ---------------------------------------------------------------------------


def test_reinforce_claim(store: ClaimsStore) -> None:
    cid = store.add_claim(SAMPLE_CLAIM)
    store.reinforce_claim(cid)
    with store._connect() as conn:
        row = conn.execute(
            "SELECT reinforcement_count FROM claims WHERE id = ?", (cid,)
        ).fetchone()
    assert row["reinforcement_count"] == 1


# ---------------------------------------------------------------------------
# contradict_claim
# ---------------------------------------------------------------------------


def test_contradict_claim(store: ClaimsStore) -> None:
    cid_a = store.add_claim("Daniel lives in Framingham, Massachusetts.")
    cid_b = store.add_claim("Daniel lives in New York City.")
    store.contradict_claim(cid_a, by_claim_id=cid_b, detail="newer info")

    with store._connect() as conn:
        row = conn.execute(
            "SELECT contradicted_by FROM claims WHERE id = ?", (cid_a,)
        ).fetchone()
    assert row["contradicted_by"] == cid_b


def test_contradict_creates_event(store: ClaimsStore) -> None:
    cid_a = store.add_claim("Daniel lives in Framingham, Massachusetts.")
    cid_b = store.add_claim("Daniel lives in New York City.")
    store.contradict_claim(cid_a, by_claim_id=cid_b)

    with store._connect() as conn:
        events = conn.execute(
            "SELECT event_type FROM claim_events WHERE claim_id = ? AND event_type = 'contradicted'",
            (cid_a,),
        ).fetchall()
    assert len(events) == 1


# ---------------------------------------------------------------------------
# expire_claim
# ---------------------------------------------------------------------------


def test_expire_claim(store: ClaimsStore) -> None:
    cid = store.add_claim(SAMPLE_CLAIM)
    store.expire_claim(cid)

    with store._connect() as conn:
        row = conn.execute(
            "SELECT valid_until FROM claims WHERE id = ?", (cid,)
        ).fetchone()
    assert row["valid_until"] is not None


# ---------------------------------------------------------------------------
# query_claims
# ---------------------------------------------------------------------------


def test_query_claims_empty_store(store: ClaimsStore) -> None:
    results = store.query_claims("anything")
    assert results == []


def test_query_claims_empty_text(store: ClaimsStore) -> None:
    store.add_claim(SAMPLE_CLAIM)
    results = store.query_claims("")
    assert results == []


def test_query_claims_excludes_expired(store: ClaimsStore) -> None:
    cid = store.add_claim(SAMPLE_CLAIM, category="relationships")
    store.expire_claim(cid)
    results = store.query_claims("Daniel wife Danielle", threshold=0.0)
    ids = [r["id"] for r in results]
    assert cid not in ids


def test_query_claims_excludes_contradicted_by_default(store: ClaimsStore) -> None:
    cid_a = store.add_claim("Daniel lives in Framingham, Massachusetts.")
    cid_b = store.add_claim("Daniel lives in New York City.")
    store.contradict_claim(cid_a, by_claim_id=cid_b)
    results = store.query_claims("where does Daniel live", threshold=0.0)
    ids = [r["id"] for r in results]
    assert cid_a not in ids
    # Non-contradicted claim should still appear
    assert cid_b in ids


def test_query_claims_includes_contradicted_when_requested(store: ClaimsStore) -> None:
    cid_a = store.add_claim("Daniel lives in Framingham, Massachusetts.")
    cid_b = store.add_claim("Daniel lives in New York City.")
    store.contradict_claim(cid_a, by_claim_id=cid_b)
    results = store.query_claims(
        "where does Daniel live", threshold=0.0, include_contradicted=True
    )
    ids = [r["id"] for r in results]
    assert cid_a in ids


def test_query_claims_returns_trust_score(store: ClaimsStore) -> None:
    store.add_claim(SAMPLE_CLAIM, category="relationships", confidence=0.9)
    results = store.query_claims("Danielle wife Daniel", threshold=0.0)
    assert results
    assert "_trust" in results[0]
    assert results[0]["_trust"] > 0


def test_query_claims_sorted_by_trust(store: ClaimsStore) -> None:
    """Highest-confidence claim should rank above low-confidence one."""
    store.add_claim("Daniel was born on June 20, 1989.", confidence=0.5)
    high = store.add_claim("Daniel's wife is Danielle Smith.", confidence=1.0)
    # Reinforce the high-confidence claim several times to push its trust up
    for _ in range(3):
        store.reinforce_claim(high)
    results = store.query_claims("Daniel personal info", threshold=0.0)
    assert results
    assert results[0]["id"] == high


def test_query_claims_n_results_limit(store: ClaimsStore) -> None:
    for i in range(10):
        store.add_claim(f"Daniel's friend number {i} lives nearby in the city.", confidence=1.0)
    results = store.query_claims("Daniel friend city", n_results=3, threshold=0.0)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# decay_report
# ---------------------------------------------------------------------------


def test_decay_report_returns_list(store: ClaimsStore) -> None:
    store.add_claim(SAMPLE_CLAIM)
    report = store.decay_report()
    assert isinstance(report, list)


def test_decay_report_excludes_recent(store: ClaimsStore) -> None:
    store.add_claim(SAMPLE_CLAIM)
    report = store.decay_report(threshold_days=30)
    # Claim was just created so should NOT be in the stale list
    assert not any(r["text"] == SAMPLE_CLAIM for r in report)


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------


def test_timeline_returns_events(store: ClaimsStore) -> None:
    store.add_claim(SAMPLE_CLAIM)
    events = store.timeline()
    assert len(events) >= 1
    assert events[0]["event_type"] == "created"


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


def test_count_totals(store: ClaimsStore) -> None:
    counts = store.count()
    assert "total" in counts
    assert "stale" in counts
    assert "contradicted" in counts
    assert counts["total"] == 0


def test_count_increments(store: ClaimsStore) -> None:
    store.add_claim(SAMPLE_CLAIM)
    counts = store.count()
    assert counts["total"] == 1


def test_count_contradicted(store: ClaimsStore) -> None:
    cid_a = store.add_claim("Daniel lives in Framingham, Massachusetts.")
    cid_b = store.add_claim("Daniel lives in New York City.")
    store.contradict_claim(cid_a, by_claim_id=cid_b)
    counts = store.count()
    assert counts["contradicted"] == 1


# ---------------------------------------------------------------------------
# import_facts
# ---------------------------------------------------------------------------


def test_import_facts_adds_new(store: ClaimsStore) -> None:
    facts = [
        {"fact": "Daniel works at TJX Companies as a Data Scientist.", "category": "work"},
        {"fact": "Daniel was born on June 20, 1989.", "category": "biographical_facts"},
    ]
    added = store.import_facts(facts)
    assert added == 2


def test_import_facts_skips_duplicates(store: ClaimsStore) -> None:
    facts = [{"fact": "Daniel was born on June 20, 1989.", "category": "biographical_facts"}]
    store.import_facts(facts)
    added_again = store.import_facts(facts)
    assert added_again == 0


def test_import_facts_skips_short(store: ClaimsStore) -> None:
    facts = [{"fact": "Hi", "category": "general"}]
    added = store.import_facts(facts)
    assert added == 0


# ---------------------------------------------------------------------------
# _claim_id helper
# ---------------------------------------------------------------------------


def test_claim_id_deterministic() -> None:
    assert _claim_id("Daniel's wife is Danielle.") == _claim_id("Daniel's wife is Danielle.")


def test_claim_id_normalisation() -> None:
    assert _claim_id("Daniel's   wife  is  Danielle.") == _claim_id("daniel's wife is danielle.")


def test_claim_id_length() -> None:
    assert len(_claim_id("any text")) == 32


# ---------------------------------------------------------------------------
# _trust_score helper
# ---------------------------------------------------------------------------


def test_trust_score_new_claim() -> None:
    score = _trust_score(confidence=1.0, reinforcement_count=0, last_reinforced_at="2099-01-01T00:00:00")
    # confidence=1, rc=0, recency≈1 → score = 1.0 * (1 + log(1)) * 1 = 1.0
    assert abs(score - 1.0) < 1e-6


def test_trust_score_decays_with_age() -> None:
    recent = _trust_score(1.0, 0, "2099-01-01T00:00:00")
    old = _trust_score(1.0, 0, "2000-01-01T00:00:00")
    assert recent > old


def test_trust_score_increases_with_reinforcement() -> None:
    low = _trust_score(1.0, 0, "2099-01-01T00:00:00")
    high = _trust_score(1.0, 5, "2099-01-01T00:00:00")
    assert high > low
