"""
Standalone embedding utility for Orion — Phase 2.2.

Provides an ``Embedder`` class that wraps ``sentence-transformers`` directly,
independent of ChromaDB.  This lets other modules (e.g. the memory extractor,
Phase 2.4 query path) obtain raw float vectors without going through a
ChromaDB collection.

The model is loaded lazily on first call and cached for the lifetime of the
instance so subsequent ``embed()`` calls are fast.

Design notes
------------
* Runs on ``device="cpu"`` — GPU is reserved for Ollama LLM inference.
* Default model ``all-MiniLM-L6-v2`` (22 MB) matches the model used by
  ``MemoryStore`` so the two share the same embedding space — distances
  computed by ChromaDB and distances computed here are directly comparable.
* Thread-safe: a module-level lock prevents concurrent model loads.
"""

from __future__ import annotations

import threading
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "all-MiniLM-L6-v2"
_DEVICE: str = "cpu"

# ---------------------------------------------------------------------------
# Lazy model cache — shared across instances that use the same model name
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


def _load_model(model_name: str):
    """Return a cached SentenceTransformer for *model_name*, loading if needed."""
    if model_name not in _model_cache:
        with _cache_lock:
            # Double-checked locking
            if model_name not in _model_cache:
                from sentence_transformers import SentenceTransformer
                _model_cache[model_name] = SentenceTransformer(
                    model_name, device=_DEVICE
                )
    return _model_cache[model_name]


# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------


class Embedder:
    """Thin wrapper around a sentence-transformers model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to ``all-MiniLM-L6-v2``
        which is already cached locally from Phase 2.1.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        # Trigger lazy load at construction time so callers can detect
        # model-not-found errors early rather than on the first embed() call.
        _load_model(model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*.

        Args:
            text: The string to embed.  Must be non-empty.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty string.")

        model = _load_model(self._model_name)
        # encode() returns a numpy array; convert to plain Python list so
        # callers don't need to worry about numpy as a dependency.
        vector = model.encode(text, device=_DEVICE, show_progress_bar=False)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of strings.

        Empty / whitespace-only strings are rejected; raise ValueError on the
        first offender so callers get a clear error.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float-vector lists, one per input string.
        """
        if not texts:
            raise ValueError("texts list must not be empty.")
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"texts[{i}] is empty or whitespace-only.")

        model = _load_model(self._model_name)
        vectors = model.encode(texts, device=_DEVICE, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    @property
    def model_name(self) -> str:
        """The sentence-transformers model name used by this embedder."""
        return self._model_name

    @property
    def device(self) -> str:
        """The compute device used for inference (always 'cpu')."""
        return _DEVICE
