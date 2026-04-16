"""
Tests for the OllamaClient.

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_llm.py -v

Requires Ollama to be running locally with phi4-mini pulled.
"""

import pytest
from src.llm.client import OllamaClient, trim_history


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    c = OllamaClient()
    if not c.is_available():
        pytest.skip("Ollama server not reachable at localhost:11434")
    return c


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def test_ollama_is_available(client):
    assert client.is_available() is True


# ---------------------------------------------------------------------------
# Non-streaming round-trip
# ---------------------------------------------------------------------------

def test_blocking_response_is_non_empty(client):
    messages = [{"role": "user", "content": "Reply with the single word: hello"}]
    response = client.chat(messages, stream=False)
    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_response_contains_expected_word(client):
    """Ask for a specific reply and check the model honours it."""
    messages = [{"role": "user", "content": "Reply with exactly the word CONFIRMED and nothing else."}]
    response = client.chat(messages, stream=False)
    assert "CONFIRMED" in response.upper()


# ---------------------------------------------------------------------------
# Streaming round-trip
# ---------------------------------------------------------------------------

def test_streaming_response_is_non_empty(client, capsys):
    messages = [{"role": "user", "content": "Say the word: streaming"}]
    response = client.chat(messages, stream=True)
    assert isinstance(response, str)
    assert len(response.strip()) > 0
    # Tokens should have been printed to stdout
    captured = capsys.readouterr()
    assert len(captured.out.strip()) > 0


# ---------------------------------------------------------------------------
# Context window trimming
# ---------------------------------------------------------------------------

def test_trim_history_removes_oldest_pairs():
    msgs = [
        {"role": "user", "content": "A" * 500},
        {"role": "assistant", "content": "B" * 500},
        {"role": "user", "content": "C" * 500},
        {"role": "assistant", "content": "D" * 500},
        {"role": "user", "content": "latest question"},
    ]
    trimmed = trim_history(msgs, limit_chars=1200)
    # Oldest pair should have been dropped; latest message must still be present
    assert trimmed[-1]["content"] == "latest question"
    assert len(trimmed) < len(msgs)


def test_trim_history_keeps_short_history():
    msgs = [
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": "reply"},
    ]
    trimmed = trim_history(msgs, limit_chars=10000)
    assert trimmed == msgs


def test_trim_history_always_keeps_last_message():
    msgs = [{"role": "user", "content": "only message"}]
    trimmed = trim_history(msgs, limit_chars=1)
    assert len(trimmed) == 1
    assert trimmed[0]["content"] == "only message"
