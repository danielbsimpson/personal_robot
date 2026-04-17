"""
Memory storage policy — governs what gets stored in short-term vs. long-term memory.

Provides two public utilities:

  * ``is_filler_message(text)`` — True when a message is too short or generic to
    contribute durable context to the history buffer.

  * ``should_store(candidate, memory_store, confidence)`` — four-gate decision
    function that returns ``(store: bool, reason: str)`` for any candidate fact.

Both are called synchronously (no LLM roundtrip) so they never block the
response path.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Long-term memory categories
# ---------------------------------------------------------------------------

LONG_TERM_CATEGORIES: frozenset[str] = frozenset(
    {
        "user_preferences",
        "biographical_facts",
        "relationships",
        "domain_expertise",
        "project_context",
    }
)

# Minimum confidence for a candidate to be written to long-term storage.
# Below this it is kept in session state only ("short_term_only").
CONFIDENCE_THRESHOLD: float = 0.8

# ---------------------------------------------------------------------------
# Transient-signal patterns — facts matching these are short-term only
# ---------------------------------------------------------------------------

_TRANSIENT_PATTERNS: list[str] = [
    r"\btoday\b",
    r"\bright now\b",
    r"\bcurrently\b",
    r"\bthis morning\b",
    r"\btonight\b",
    r"\bthis week\b",
    r"\byesterday\b",
    r"\btomorrow\b",
    r"\bweather\b",
    r"\btemperature\b",
    r"\btraffic\b",
]

# ---------------------------------------------------------------------------
# Filler detection
# ---------------------------------------------------------------------------

# Exact normalised phrases considered pure conversational filler.
_FILLER_PHRASES: frozenset[str] = frozenset(
    {
        "ok",
        "okay",
        "sure",
        "yes",
        "no",
        "nope",
        "yep",
        "yeah",
        "thanks",
        "thank you",
        "thx",
        "ty",
        "got it",
        "alright",
        "cool",
        "nice",
        "great",
        "awesome",
        "good",
        "fine",
        "k",
        "lol",
        "haha",
        "ha",
        "hmm",
        "hm",
        "ah",
        "oh",
        "uh",
        "um",
    }
)

# Messages shorter than this (after stripping) are always considered filler.
_MIN_CONTENT_LENGTH: int = 10


def is_filler_message(text: str) -> bool:
    """Return True if a message is too short or generic to contribute durable context.

    Filler messages (e.g. "ok", "sure", "got it") clutter the history buffer
    without adding information.  Filtering them out keeps the recency window
    focused on substantive exchanges.

    Args:
        text: The raw message content string.

    Returns:
        True when the message should NOT be retained in the history buffer.
    """
    stripped = text.strip()
    if len(stripped) < _MIN_CONTENT_LENGTH:
        return True
    return stripped.lower() in _FILLER_PHRASES


# ---------------------------------------------------------------------------
# Internal helpers for should_store
# ---------------------------------------------------------------------------

# Keyword signals that suggest a fact belongs in a long-term category.
_CATEGORY_SIGNALS: tuple[str, ...] = (
    # user_preferences
    "prefer",
    "like ",
    "likes",
    "loves",
    "hate",
    "enjoy",
    "dislike",
    "favourite",
    "favorite",
    "always",
    "never",
    # biographical_facts
    "born",
    "grew up",
    "lives",
    "works",
    "studied",
    "degree",
    "job",
    "profession",
    "age",
    "family",
    "sister",
    "brother",
    "mother",
    "father",
    "parent",
    "husband",
    "wife",
    "partner",
    # project_context
    "project",
    "working on",
    "building",
    "coding",
    "writing",
    "planning",
    # domain_expertise
    "expert",
    "experienced",
    "knows",
    "specialises",
    "specializes",
    "background in",
    # relationships
    "friend",
    "colleague",
    "boss",
    "manager",
    "mentor",
)


def _is_repeat(candidate: str, memory_store: dict) -> bool:
    """Return True if the candidate is semantically present in the memory store.

    Uses a simple substring check keyed off the normalised candidate text.
    A proper novelty check would embed both strings and compare cosine distance;
    this heuristic is intentionally cheap and fast.
    """
    needle = candidate.lower().strip()
    for value in memory_store.values():
        if isinstance(value, str) and needle in value.lower():
            return True
        if isinstance(value, dict):
            for v in value.values():
                if isinstance(v, str) and needle in v.lower():
                    return True
    return False


def _fits_category(candidate: str) -> bool:
    """Return True if the candidate plausibly belongs in a long-term category.

    Uses a keyword heuristic — no LLM call — so it runs synchronously.
    """
    lower = candidate.lower()
    return any(signal in lower for signal in _CATEGORY_SIGNALS)


def _is_transient(candidate: str) -> bool:
    """Return True if the candidate contains signals that the fact is time-limited."""
    lower = candidate.lower()
    return any(re.search(pattern, lower) for pattern in _TRANSIENT_PATTERNS)


# ---------------------------------------------------------------------------
# Public: should_store
# ---------------------------------------------------------------------------


def should_store(
    candidate: str,
    memory_store: dict,
    confidence: float,
) -> tuple[bool, str]:
    """Decide whether a candidate fact should be persisted to long-term memory.

    The decision follows a four-step gate:

    1. **Repeat check** — already present in store → skip entirely.
    2. **Category check** — does not fit any long-term category → "incidental".
    3. **Transience check** — time-limited signal → "short_term_only".
    4. **Confidence gate** — below threshold → "short_term_only".

    Args:
        candidate:    Plain-text statement of the fact to evaluate.
        memory_store: Current in-memory fact store (dict-of-dicts, e.g. soul data).
        confidence:   Extraction model's self-rated certainty in [0.0, 1.0].

    Returns:
        A ``(store: bool, reason: str)`` tuple where *reason* is one of:
        ``"long_term"``, ``"short_term_only"``, ``"repeat"``, ``"incidental"``.
    """
    # Gate 1 — already stored?
    if _is_repeat(candidate, memory_store):
        return False, "repeat"

    # Gate 2 — fits a long-term category?
    if not _fits_category(candidate):
        return False, "incidental"

    # Gate 3 — transient information?
    if _is_transient(candidate):
        return True, "short_term_only"

    # Gate 4 — confidence threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        return True, "long_term"

    return True, "short_term_only"
