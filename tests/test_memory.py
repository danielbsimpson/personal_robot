"""
Tests for MemoryStore (Phase 2.1 — ChromaDB RAG store).

All tests use a temporary directory for ChromaDB persistence so the
production data/memory/ directory is never touched.

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_memory.py -v
"""

import pytest
from pathlib import Path
from src.memory.vector_store import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    """A fresh MemoryStore backed by a temp directory for each test."""
    return MemoryStore(persist_dir=tmp_path / "memory")


# ---------------------------------------------------------------------------
# add_memory
# ---------------------------------------------------------------------------

def test_add_memory_returns_doc_id(store: MemoryStore) -> None:
    doc_id = store.add_memory("Daniel enjoys jazz music.", {"source": "test"})
    assert isinstance(doc_id, str)
    assert len(doc_id) == 16


def test_add_memory_increments_count(store: MemoryStore) -> None:
    assert store.count() == 0
    store.add_memory("Daniel works at TJX.")
    assert store.count() == 1
    store.add_memory("Danielle is an artist.")
    assert store.count() == 2


def test_add_memory_is_idempotent(store: MemoryStore) -> None:
    """Re-adding identical text should not increase the count (upsert by hash)."""
    text = "Orion is a personal robot."
    store.add_memory(text)
    store.add_memory(text)
    assert store.count() == 1


def test_add_memory_rejects_empty_string(store: MemoryStore) -> None:
    with pytest.raises(ValueError):
        store.add_memory("")


def test_add_memory_rejects_whitespace_only(store: MemoryStore) -> None:
    with pytest.raises(ValueError):
        store.add_memory("   ")


# ---------------------------------------------------------------------------
# query_memory
# ---------------------------------------------------------------------------

def test_query_returns_relevant_result(store: MemoryStore) -> None:
    store.add_memory("Daniel's favourite band is Magnolia Park.")
    results = store.query_memory("What music does Daniel like?", threshold=0.0)
    assert len(results) == 1
    assert "Magnolia Park" in results[0]


def test_query_returns_empty_when_store_is_empty(store: MemoryStore) -> None:
    results = store.query_memory("anything")
    assert results == []


def test_query_returns_empty_for_blank_text(store: MemoryStore) -> None:
    store.add_memory("Some memory.")
    assert store.query_memory("") == []
    assert store.query_memory("   ") == []


def test_query_filters_below_threshold(store: MemoryStore) -> None:
    """With threshold=1.0 nothing should match (no document is a perfect duplicate)."""
    store.add_memory("Daniel enjoys hiking in New England.")
    results = store.query_memory("What does Daniel like to do?", threshold=1.0)
    assert results == []


def test_query_respects_n_results(store: MemoryStore) -> None:
    for i in range(5):
        store.add_memory(f"Memory entry number {i} about Daniel's hobbies.")
    results = store.query_memory("Daniel's hobbies", n_results=2, threshold=0.0)
    assert len(results) <= 2


def test_query_semantic_relevance(store: MemoryStore) -> None:
    """A semantically relevant memory should score above an irrelevant one."""
    store.add_memory("Daniel is a data scientist at TJX Companies.")
    store.add_memory("The sky is blue and clouds are white.")
    results = store.query_memory("What is Daniel's job?", threshold=0.2)
    # The data-science memory should be the top hit
    assert any("data scientist" in r or "TJX" in r for r in results)


# ---------------------------------------------------------------------------
# clear_memory
# ---------------------------------------------------------------------------

def test_clear_memory_empties_store(store: MemoryStore) -> None:
    store.add_memory("Memory A")
    store.add_memory("Memory B")
    assert store.count() == 2
    store.clear_memory()
    assert store.count() == 0


def test_clear_memory_allows_subsequent_adds(store: MemoryStore) -> None:
    store.add_memory("Old memory.")
    store.clear_memory()
    store.add_memory("New memory after clear.")
    assert store.count() == 1


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_persistence_across_instances(tmp_path: Path) -> None:
    """Data written by one MemoryStore instance should survive a new instance."""
    mem_dir = tmp_path / "memory"
    store1 = MemoryStore(persist_dir=mem_dir)
    store1.add_memory("Persistent fact: Daniel grew up in West Virginia.")

    store2 = MemoryStore(persist_dir=mem_dir)
    assert store2.count() == 1
    results = store2.query_memory("Where did Daniel grow up?", threshold=0.0)
    assert any("West Virginia" in r for r in results)
