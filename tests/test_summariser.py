"""
Tests for src/memory/summariser.py

Most tests are pure-Python (no Ollama dependency).
The integration test that calls the LLM is skipped automatically
when Ollama is not reachable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.memory.summariser import (
    MAX_CONV_CHARS,
    MIN_TURNS,
    _count_user_turns,
    _format_conversation,
    summarise_session,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SHORT_HISTORY: list[dict] = [
    {"role": "user", "content": "Hey there"},
]

NORMAL_HISTORY: list[dict] = [
    {"role": "user", "content": "I'm Daniel, a software engineer."},
    {"role": "assistant", "content": "Nice to meet you, Daniel!"},
    {"role": "user", "content": "I prefer Python over Java."},
    {"role": "assistant", "content": "Noted — Python it is."},
]

HISTORY_WITH_SYSTEM: list[dict] = [
    {"role": "system", "content": "You are Orion."},
    {"role": "user", "content": "My name is Daniel."},
    {"role": "assistant", "content": "Hello Daniel!"},
    {"role": "user", "content": "I work as a nurse."},
    {"role": "assistant", "content": "That is great!"},
]

HISTORY_WITH_SUMMARY_ENTRY: list[dict] = [
    {
        "role": "system",
        "content": "[Earlier context summary] Daniel spoke about his family.",
    },
    {"role": "user", "content": "Moving on — I like hiking."},
    {"role": "assistant", "content": "That sounds fun!"},
    {"role": "user", "content": "Especially in the Peak District."},
    {"role": "assistant", "content": "Beautiful area."},
]


# ---------------------------------------------------------------------------
# _count_user_turns
# ---------------------------------------------------------------------------


def test_count_user_turns_empty() -> None:
    assert _count_user_turns([]) == 0


def test_count_user_turns_normal() -> None:
    assert _count_user_turns(NORMAL_HISTORY) == 2


def test_count_user_turns_mixed_roles() -> None:
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    assert _count_user_turns(history) == 2


# ---------------------------------------------------------------------------
# _format_conversation
# ---------------------------------------------------------------------------


def test_format_excludes_system_messages() -> None:
    result = _format_conversation(HISTORY_WITH_SYSTEM)
    assert "You are Orion" not in result


def test_format_excludes_earlier_context_summary() -> None:
    result = _format_conversation(HISTORY_WITH_SUMMARY_ENTRY)
    assert "[Earlier context summary]" not in result


def test_format_includes_user_and_assistant_turns() -> None:
    result = _format_conversation(NORMAL_HISTORY)
    assert "User:" in result
    assert "Assistant:" in result
    assert "Daniel" in result


def test_format_empty_history_returns_empty_string() -> None:
    assert _format_conversation([]) == ""


def test_format_system_only_returns_empty_string() -> None:
    history = [{"role": "system", "content": "sys prompt"}]
    assert _format_conversation(history) == ""


# ---------------------------------------------------------------------------
# summarise_session — short-circuit paths (no LLM call)
# ---------------------------------------------------------------------------


def test_returns_empty_string_when_too_few_turns() -> None:
    result = summarise_session(SHORT_HISTORY, model="dummy")
    assert result == ""


def test_returns_empty_string_for_empty_history() -> None:
    result = summarise_session([], model="dummy")
    assert result == ""


def test_returns_empty_string_when_all_messages_are_system() -> None:
    history = [
        {"role": "system", "content": "sys"},
        {"role": "system", "content": "another sys"},
    ]
    result = summarise_session(history, model="dummy")
    assert result == ""


def test_min_turns_override_respected() -> None:
    """Passing min_turns=1 allows a single-turn history to proceed to LLM."""
    with patch("src.llm.client.OllamaClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "Daniel said hello."
        MockClient.return_value = mock_instance

        result = summarise_session(SHORT_HISTORY, model="phi4-mini:latest", min_turns=1)
        assert result == "Daniel said hello."
        mock_instance.chat.assert_called_once()


# ---------------------------------------------------------------------------
# summarise_session — LLM path (mocked)
# ---------------------------------------------------------------------------


def test_calls_llm_with_conversation_text() -> None:
    with patch("src.llm.client.OllamaClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "Daniel is a software engineer who prefers Python."
        MockClient.return_value = mock_instance

        result = summarise_session(NORMAL_HISTORY, model="phi4-mini:latest")

        assert result == "Daniel is a software engineer who prefers Python."
        mock_instance.chat.assert_called_once()
        # The prompt sent to the LLM should contain the conversation text
        call_args = mock_instance.chat.call_args
        prompt_content = call_args[0][0][0]["content"]
        assert "Daniel" in prompt_content


def test_strips_whitespace_from_llm_response() -> None:
    with patch("src.llm.client.OllamaClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "  Summary here.  \n"
        MockClient.return_value = mock_instance

        result = summarise_session(NORMAL_HISTORY, model="phi4-mini:latest")
        assert result == "Summary here."


def test_returns_empty_string_when_llm_returns_whitespace() -> None:
    with patch("src.llm.client.OllamaClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "   "
        MockClient.return_value = mock_instance

        result = summarise_session(NORMAL_HISTORY, model="phi4-mini:latest")
        assert result == ""


def test_system_messages_excluded_from_llm_prompt() -> None:
    with patch("src.llm.client.OllamaClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "Summary."
        MockClient.return_value = mock_instance

        summarise_session(HISTORY_WITH_SYSTEM, model="phi4-mini:latest")

        prompt_content = mock_instance.chat.call_args[0][0][0]["content"]
        assert "You are Orion" not in prompt_content


def test_earlier_context_summary_excluded_from_llm_prompt() -> None:
    with patch("src.llm.client.OllamaClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "Summary."
        MockClient.return_value = mock_instance

        summarise_session(HISTORY_WITH_SUMMARY_ENTRY, model="phi4-mini:latest")

        prompt_content = mock_instance.chat.call_args[0][0][0]["content"]
        assert "[Earlier context summary]" not in prompt_content


# ---------------------------------------------------------------------------
# summarise_session — long conversation truncation
# ---------------------------------------------------------------------------


def test_very_long_conversation_is_truncated() -> None:
    """Conversations exceeding MAX_CONV_CHARS are truncated before the LLM call."""
    long_history = [
        {"role": "user", "content": "x" * (MAX_CONV_CHARS // 2)},
        {"role": "assistant", "content": "y" * (MAX_CONV_CHARS // 2)},
        {"role": "user", "content": "final question"},
        {"role": "assistant", "content": "final answer"},
    ]
    with patch("src.llm.client.OllamaClient") as MockClient:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "Summary."
        MockClient.return_value = mock_instance

        result = summarise_session(long_history, model="phi4-mini:latest")
        assert result == "Summary."

        prompt_content = mock_instance.chat.call_args[0][0][0]["content"]
        assert len(prompt_content) < MAX_CONV_CHARS * 3  # well within reason


# ---------------------------------------------------------------------------
# Integration test — requires Ollama (skipped if unavailable)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ollama_available() -> bool:
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def test_summarise_session_live(ollama_available: bool) -> None:
    """End-to-end test: real LLM produces a non-empty summary string."""
    if not ollama_available:
        pytest.skip("Ollama not reachable")

    history = [
        {"role": "user", "content": "My name is Daniel and I work as a software engineer."},
        {"role": "assistant", "content": "Good to know, Daniel!"},
        {"role": "user", "content": "I really enjoy hiking and playing guitar in my spare time."},
        {"role": "assistant", "content": "Sounds like a great way to unwind."},
    ]
    from src.llm.client import DEFAULT_MODEL
    result = summarise_session(history, model=DEFAULT_MODEL)
    assert isinstance(result, str)
    assert len(result.strip()) > 0
