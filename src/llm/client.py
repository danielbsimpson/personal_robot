"""
Ollama HTTP API client for local LLM inference.

Wraps the /api/chat endpoint with streaming support, conversation history
management, and a configurable system prompt.
"""

import json
import logging
import sys
from collections.abc import Iterator
from typing import Optional

import requests

from src.llm.prompts import BASE_SYSTEM_PROMPT

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "phi4-mini"
# Rough token budget before trimming oldest messages (1 token ≈ 4 chars)
DEFAULT_CONTEXT_LIMIT = 4096

# ---------------------------------------------------------------------------
# Lazy trim logger — writes to data/logs/context_trim.log
# ---------------------------------------------------------------------------

_trim_log: Optional[logging.Logger] = None


def _get_trim_log() -> logging.Logger:
    """Return the context_trim logger, creating it on first call."""
    global _trim_log
    if _trim_log is None:
        from src.utils.log import get_logger
        _trim_log = get_logger("context_trim")
    return _trim_log


class OllamaClient:
    """Thin wrapper around the Ollama /api/chat endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str = BASE_SYSTEM_PROMPT,
        context_limit: int = DEFAULT_CONTEXT_LIMIT,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.context_limit = context_limit
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict], stream: bool = True) -> str:
        """Send a messages list to the LLM and return the full response text.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
                      The system prompt is prepended automatically.
            stream:   If True, tokens are printed to stdout as they arrive.

        Returns:
            The complete assistant response as a single string.
        """
        payload = self._build_payload(messages, stream=stream)

        if stream:
            return self._stream(payload)
        else:
            return self._blocking(payload)

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(self, messages: list[dict], stream: bool) -> dict:
        """Prepend system prompt and return the full request payload."""
        system_message = {"role": "system", "content": self.system_prompt}
        full_messages = [system_message] + messages
        return {
            "model": self.model,
            "messages": full_messages,
            "stream": stream,
        }

    def _stream(self, payload: dict) -> str:
        """POST to /api/chat with streaming=True, print tokens, return full text."""
        url = f"{self.base_url}/api/chat"
        full_response = []

        with self._session.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_response.append(token)
                if chunk.get("done"):
                    break

        print()  # newline after streaming finishes
        return "".join(full_response)

    def _blocking(self, payload: dict) -> str:
        """POST to /api/chat with streaming=False, return full text."""
        url = f"{self.base_url}/api/chat"
        resp = self._session.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ------------------------------------------------------------------
# Context window trimming utility
# ------------------------------------------------------------------

def trim_history(
    messages: list[dict],
    limit_chars: int = DEFAULT_CONTEXT_LIMIT * 4,
    budget_chars: Optional[int] = None,
) -> list[dict]:
    """Drop oldest user/assistant pairs until the history fits within the budget.

    The most recent message is always kept. Trims from the front in pairs
    (user + assistant) to preserve conversational coherence.

    Args:
        messages:     List of {"role": ..., "content": ...} dicts (no system message).
        limit_chars:  Legacy cap — used when *budget_chars* is not supplied.
        budget_chars: Explicit character budget from ``ContextBudget``; overrides
                      *limit_chars* when provided. Backwards-compatible: if omitted
                      the existing ``DEFAULT_CONTEXT_LIMIT * 4`` default is used.

    Returns:
        Trimmed messages list.
    """
    cap = budget_chars if budget_chars is not None else limit_chars
    original_count = len(messages)
    original_chars = sum(len(m["content"]) for m in messages)
    dropped: list[dict] = []

    while True:
        total = sum(len(m["content"]) for m in messages)
        if total <= cap or len(messages) <= 2:
            break
        dropped.extend(messages[:2])
        messages = messages[2:]

    if dropped:
        _get_trim_log().info(
            "trimmed %d\u2192%d messages (%d\u2192%d chars) | dropped: %s",
            original_count,
            len(messages),
            original_chars,
            sum(len(m["content"]) for m in messages),
            [{"role": m["role"], "preview": m["content"][:120]} for m in dropped],
        )

    return messages
