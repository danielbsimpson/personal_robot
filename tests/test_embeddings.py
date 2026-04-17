"""
Tests for src/memory/embeddings.py

All tests run on the cached all-MiniLM-L6-v2 model (already downloaded in
Phase 2.1) so no network access is required.
"""

import math

import pytest

from src.memory.embeddings import DEFAULT_MODEL, Embedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Return a single Embedder instance shared across the test module."""
    return Embedder()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_model_name(embedder: Embedder) -> None:
    assert embedder.model_name == DEFAULT_MODEL


def test_device_is_cpu(embedder: Embedder) -> None:
    assert embedder.device == "cpu"


# ---------------------------------------------------------------------------
# embed() — basic contract
# ---------------------------------------------------------------------------


def test_embed_returns_list_of_floats(embedder: Embedder) -> None:
    vec = embedder.embed("Hello world")
    assert isinstance(vec, list)
    assert all(isinstance(x, float) for x in vec)


def test_embed_vector_has_expected_dimension(embedder: Embedder) -> None:
    # all-MiniLM-L6-v2 produces 384-dimensional vectors
    vec = embedder.embed("Test sentence")
    assert len(vec) == 384


def test_embed_same_text_returns_identical_vector(embedder: Embedder) -> None:
    v1 = embedder.embed("Daniel prefers Python")
    v2 = embedder.embed("Daniel prefers Python")
    assert v1 == v2


def test_embed_different_texts_return_different_vectors(embedder: Embedder) -> None:
    v1 = embedder.embed("Daniel loves hiking")
    v2 = embedder.embed("The cat sat on the mat")
    assert v1 != v2


# ---------------------------------------------------------------------------
# embed() — input validation
# ---------------------------------------------------------------------------


def test_embed_raises_on_empty_string(embedder: Embedder) -> None:
    with pytest.raises(ValueError):
        embedder.embed("")


def test_embed_raises_on_whitespace_only(embedder: Embedder) -> None:
    with pytest.raises(ValueError):
        embedder.embed("   ")


# ---------------------------------------------------------------------------
# embed() — semantic properties
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def test_similar_sentences_have_higher_similarity_than_unrelated(
    embedder: Embedder,
) -> None:
    v_music = embedder.embed("Daniel enjoys listening to jazz music")
    v_jazz = embedder.embed("Daniel likes jazz and plays guitar")
    v_unrelated = embedder.embed("The quarterly sales figures were down ten percent")

    sim_related = _cosine_similarity(v_music, v_jazz)
    sim_unrelated = _cosine_similarity(v_music, v_unrelated)
    assert sim_related > sim_unrelated


# ---------------------------------------------------------------------------
# embed_batch()
# ---------------------------------------------------------------------------


def test_embed_batch_returns_correct_count(embedder: Embedder) -> None:
    texts = ["first sentence", "second sentence", "third sentence"]
    results = embedder.embed_batch(texts)
    assert len(results) == 3


def test_embed_batch_each_vector_has_correct_dimension(embedder: Embedder) -> None:
    texts = ["alpha", "beta"]
    results = embedder.embed_batch(texts)
    assert all(len(v) == 384 for v in results)


def test_embed_batch_matches_individual_embed(embedder: Embedder) -> None:
    texts = ["hello", "world"]
    batch_results = embedder.embed_batch(texts)
    for text, batch_vec in zip(texts, batch_results):
        single_vec = embedder.embed(text)
        # Vectors should match to floating-point precision
        assert batch_vec == pytest.approx(single_vec, abs=1e-5)


def test_embed_batch_raises_on_empty_list(embedder: Embedder) -> None:
    with pytest.raises(ValueError):
        embedder.embed_batch([])


def test_embed_batch_raises_on_empty_string_in_list(embedder: Embedder) -> None:
    with pytest.raises(ValueError):
        embedder.embed_batch(["valid sentence", ""])


# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------


def test_two_instances_share_cached_model() -> None:
    """Constructing two Embedders does not load the model twice."""
    from src.memory.embeddings import _model_cache

    before_count = len(_model_cache)
    _ = Embedder()
    _ = Embedder()
    # Cache size should not have grown — same model reused
    assert len(_model_cache) == before_count
