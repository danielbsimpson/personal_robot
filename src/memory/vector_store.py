"""
Persistent vector memory store for Orion — Phase 2 RAG.

Wraps ChromaDB (local persistence) with a sentence-transformers embedding
function so memories can be added, queried by semantic similarity, and
cleared for development/testing.

Usage
-----
    from src.memory.vector_store import MemoryStore

    store = MemoryStore()
    store.add_memory("Daniel likes jazz and plays guitar.", {"source": "session_1"})
    results = store.query_memory("What music does Daniel like?")
    # → ["Daniel likes jazz and plays guitar."]
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MEMORY_DIR = _PROJECT_ROOT / "data" / "memory"
_COLLECTION_NAME = "orion_memories"

# ---------------------------------------------------------------------------
# Lazy memory logger
# ---------------------------------------------------------------------------

_mem_log = None


def _get_mem_log():
    global _mem_log
    if _mem_log is None:
        from src.utils.log import get_logger
        _mem_log = get_logger("memory")
    return _mem_log


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """Semantic memory store backed by ChromaDB with local persistence.

    Parameters
    ----------
    persist_dir:
        Path to the directory where ChromaDB stores its data.
        Defaults to ``data/memory/``.
    embedding_model:
        Sentence-transformers model name used for embedding.
        ``all-MiniLM-L6-v2`` (22 MB) gives a good speed/quality trade-off
        and runs comfortably on CPU while the GPU is reserved for Ollama.
    similarity_threshold:
        Default cosine-similarity floor for ``query_memory()``.  Documents
        whose distance exceeds ``1 - threshold`` are filtered out so that
        irrelevant memories are never injected into the context.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.35,
    ) -> None:
        self._persist_dir = Path(persist_dir) if persist_dir else DEFAULT_MEMORY_DIR
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._threshold = similarity_threshold
        self._lock = threading.Lock()

        # Lazy imports keep startup fast when running in CLI / test mode
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
            device="cpu",  # CPU keeps inference stable; GPU reserved for Ollama
        )
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_memory(self, text: str, metadata: Optional[dict] = None) -> str:
        """Embed *text* and store it in the collection.

        Parameters
        ----------
        text:
            The memory text to store (e.g. a session summary).
        metadata:
            Optional dict of string key/value pairs attached to the document
            (e.g. ``{"source": "session_2026-04-16", "type": "summary"}``).

        Returns
        -------
        str
            The document ID assigned to this memory.
        """
        if not text or not text.strip():
            raise ValueError("Cannot store an empty memory.")

        meta = {k: str(v) for k, v in (metadata or {}).items()}
        # ChromaDB 1.5+ rejects empty metadata dicts; pass None instead
        meta_arg: Optional[dict] = meta if meta else None

        # Use a deterministic ID based on content so re-adding the same
        # summary is idempotent (ChromaDB upserts on matching IDs).
        import hashlib
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]

        with self._lock:
            self._collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[meta_arg],
            )
        _get_mem_log().info(
            "add_memory: stored doc_id=%s len=%d metadata=%s",
            doc_id,
            len(text),
            meta,
        )
        return doc_id

    def query_memory(
        self,
        text: str,
        n_results: int = 5,
        threshold: Optional[float] = None,
    ) -> list[str]:
        """Return memories semantically similar to *text*.

        Only documents whose cosine similarity meets *threshold* are returned.
        If nothing clears the threshold, an empty list is returned so the
        caller can safely skip injecting a ``## Relevant Memory`` block.

        Parameters
        ----------
        text:
            The query string (typically the user's latest message).
        n_results:
            Maximum number of results to retrieve before threshold filtering.
        threshold:
            Cosine similarity floor (0–1).  Defaults to the instance-level
            ``similarity_threshold`` passed at construction time.
        """
        if not text or not text.strip():
            return []

        floor = threshold if threshold is not None else self._threshold
        count = self._collection.count()
        if count == 0:
            return []

        actual_n = min(n_results, count)
        with self._lock:
            results = self._collection.query(
                query_texts=[text],
                n_results=actual_n,
                include=["documents", "distances"],
            )

        documents: list[str] = results["documents"][0]
        distances: list[float] = results["distances"][0]

        # ChromaDB cosine distance = 1 − similarity; convert back to similarity
        filtered = [
            doc
            for doc, dist in zip(documents, distances)
            if (1.0 - dist) >= floor
        ]
        _get_mem_log().info(
            "query_memory: query=%r n_candidates=%d threshold=%.2f matched=%d",
            text[:80],
            len(documents),
            floor,
            len(filtered),
        )
        return filtered

    def clear_memory(self) -> None:
        """Delete all stored memories. Intended for development and testing."""
        with self._lock:
            self._client.delete_collection(_COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                embedding_function=self._ef,
                metadata={"hnsw:space": "cosine"},
            )

    def count(self) -> int:
        """Return the number of memories currently stored."""
        return self._collection.count()
