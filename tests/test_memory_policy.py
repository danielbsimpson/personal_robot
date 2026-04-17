"""
Tests for src/memory/policy.py

Covers all four branches of should_store() and the is_filler_message() helper.
No external dependencies — pure Python only.
"""

import pytest

from src.memory.policy import (
    CONFIDENCE_THRESHOLD,
    LONG_TERM_CATEGORIES,
    is_filler_message,
    should_store,
)


# ---------------------------------------------------------------------------
# is_filler_message
# ---------------------------------------------------------------------------


class TestIsFillerMessage:
    def test_single_word_ok_is_filler(self):
        assert is_filler_message("ok") is True

    def test_thanks_is_filler(self):
        assert is_filler_message("thanks") is True

    def test_got_it_is_filler(self):
        assert is_filler_message("got it") is True

    def test_whitespace_only_is_filler(self):
        assert is_filler_message("   ") is True

    def test_empty_string_is_filler(self):
        assert is_filler_message("") is True

    def test_single_char_is_filler(self):
        assert is_filler_message("k") is True

    def test_substantive_message_is_not_filler(self):
        assert is_filler_message("Can you tell me about Python decorators?") is False

    def test_medium_length_sentence_is_not_filler(self):
        assert is_filler_message("I prefer concise answers please.") is False

    def test_case_insensitive_filler(self):
        # "Thanks" upper-cased should still be detected as filler
        # (strip().lower() applied before lookup)
        assert is_filler_message("THANKS") is True

    def test_longer_non_filler_with_filler_word_inside(self):
        # "sure" alone is filler, but a sentence containing it is not
        assert is_filler_message("I am not sure about that theory.") is False


# ---------------------------------------------------------------------------
# should_store — Gate 1: repeat check
# ---------------------------------------------------------------------------


class TestShouldStoreRepeat:
    def test_exact_repeat_returns_false_repeat(self):
        store = {"user": {"job": "Daniel prefers dark mode"}}
        store_flag, reason = should_store(
            "Daniel prefers dark mode", store, confidence=0.95
        )
        assert store_flag is False
        assert reason == "repeat"

    def test_substring_repeat_returns_false_repeat(self):
        store = {"user": {"name": "Daniel is a software engineer"}}
        store_flag, reason = should_store(
            "software engineer", store, confidence=0.9
        )
        assert store_flag is False
        assert reason == "repeat"

    def test_non_repeat_passes_gate_1(self):
        store = {"user": {"name": "Alice"}}
        # A completely unrelated fact should not be caught by the repeat gate.
        # (It may still fail a later gate, so we check it doesn't return "repeat".)
        _, reason = should_store("Daniel loves hiking", store, confidence=0.9)
        assert reason != "repeat"


# ---------------------------------------------------------------------------
# should_store — Gate 2: category check (incidental)
# ---------------------------------------------------------------------------


class TestShouldStoreCategory:
    def test_pure_chat_is_incidental(self):
        store: dict = {}
        store_flag, reason = should_store(
            "the sky was really nice this afternoon", store, confidence=0.9
        )
        assert store_flag is False
        assert reason == "incidental"

    def test_preference_fact_passes_category_gate(self):
        store: dict = {}
        _, reason = should_store(
            "Daniel prefers Python over Java", store, confidence=0.9
        )
        assert reason != "incidental"

    def test_biographical_fact_passes_category_gate(self):
        store: dict = {}
        _, reason = should_store(
            "Daniel grew up in Manchester", store, confidence=0.9
        )
        assert reason != "incidental"


# ---------------------------------------------------------------------------
# should_store — Gate 3: transience check
# ---------------------------------------------------------------------------


class TestShouldStoreTransient:
    def test_today_signals_short_term_only(self):
        store: dict = {}
        store_flag, reason = should_store(
            "Daniel works from home today", store, confidence=0.9
        )
        # Passes gates 1 & 2 (has "works" signal), but should be short_term_only
        assert store_flag is True
        assert reason == "short_term_only"

    def test_currently_signals_short_term_only(self):
        store: dict = {}
        store_flag, reason = should_store(
            "He is currently working on a Python project", store, confidence=0.95
        )
        assert store_flag is True
        assert reason == "short_term_only"

    def test_weather_is_short_term_only(self):
        store: dict = {}
        store_flag, reason = should_store(
            "Daniel likes the weather today", store, confidence=0.85
        )
        assert store_flag is True
        assert reason == "short_term_only"


# ---------------------------------------------------------------------------
# should_store — Gate 4: confidence threshold
# ---------------------------------------------------------------------------


class TestShouldStoreConfidence:
    def test_high_confidence_stable_fact_is_long_term(self):
        store: dict = {}
        store_flag, reason = should_store(
            "Daniel prefers dark mode for all editors", store, confidence=0.9
        )
        assert store_flag is True
        assert reason == "long_term"

    def test_low_confidence_stable_fact_is_short_term_only(self):
        store: dict = {}
        store_flag, reason = should_store(
            "Daniel might like hiking in mountains", store, confidence=0.5
        )
        assert store_flag is True
        assert reason == "short_term_only"

    def test_exactly_at_threshold_is_long_term(self):
        store: dict = {}
        store_flag, reason = should_store(
            "Daniel loves cooking Italian food", store, confidence=CONFIDENCE_THRESHOLD
        )
        assert store_flag is True
        assert reason == "long_term"

    def test_just_below_threshold_is_short_term_only(self):
        store: dict = {}
        confidence = CONFIDENCE_THRESHOLD - 0.01
        store_flag, reason = should_store(
            "Daniel enjoys board games with friends", store, confidence=confidence
        )
        assert store_flag is True
        assert reason == "short_term_only"


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_long_term_categories_is_non_empty():
    assert len(LONG_TERM_CATEGORIES) > 0


def test_confidence_threshold_in_valid_range():
    assert 0.0 < CONFIDENCE_THRESHOLD < 1.0
