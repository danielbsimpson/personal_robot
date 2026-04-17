"""
Session summariser for Orion — Phase 2.3.

Provides ``summarise_session(conversation_history, model, base_url)`` which
feeds a completed conversation to the LLM and returns a concise paragraph
suitable for storage in the long-term vector memory (Phase 2.4).

Design notes
------------
* Uses the existing ``SUMMARISE_SESSION_PROMPT`` from ``src/llm/prompts`` so
  the summarisation style is consistent with the ``compress_history()`` path.
* Filters out system messages and the ``[Earlier context summary]`` entries
  injected by ``compress_history()`` — those are trim artefacts, not new facts.
* Returns an empty string when the conversation is too short to be worth
  summarising (fewer than ``MIN_TURNS`` user+assistant exchanges), so the
  Phase 2.4 caller can skip the ``add_memory()`` call entirely rather than
  storing a trivial "no content" string.
* Stateless — no side effects; safe to call from any thread.
"""

from __future__ import annotations

from typing import Optional

# Minimum number of user messages required before a summary is generated.
# A single-message exchange carries too little signal to be worth storing.
MIN_TURNS: int = 2

# Maximum characters of conversation text forwarded to the LLM.
# For very long sessions the oldest turns are already compressed by
# compress_history(); this cap prevents an absurdly large single prompt.
MAX_CONV_CHARS: int = 12_000


def _format_conversation(history: list[dict]) -> str:
    """Convert a history list into a readable transcript string.

    Skips system messages and [Earlier context summary] entries (injected by
    ``compress_history()``) so the LLM only sees real conversational turns.

    Args:
        history: List of ``{"role": ..., "content": ...}`` dicts.

    Returns:
        A newline-separated transcript, or an empty string if no user/assistant
        turns are present.
    """
    lines: list[str] = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            continue
        if content.startswith("[Earlier context summary]"):
            continue
        label = role.capitalize()
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def _count_user_turns(history: list[dict]) -> int:
    """Return the number of user messages in *history*."""
    return sum(1 for m in history if m.get("role") == "user")


def summarise_session(
    conversation_history: list[dict],
    model: str,
    base_url: str = "http://localhost:11434",
    min_turns: int = MIN_TURNS,
) -> str:
    """Summarise a completed conversation session into a single paragraph.

    Feeds the transcript to the LLM using ``SUMMARISE_SESSION_PROMPT`` and
    returns the response text.  Returns an empty string when the conversation
    is too short or empty to warrant summarisation.

    Args:
        conversation_history:
            The full list of ``{"role": ..., "content": ...}`` dicts for the
            session (no system messages required, but harmless if present).
        model:
            Ollama model tag to use for summarisation (e.g. ``"phi4-mini:latest"``).
        base_url:
            Ollama server URL.  Defaults to ``"http://localhost:11434"``.
        min_turns:
            Minimum number of user turns required before summarising.
            Conversations shorter than this return ``""`` immediately.

    Returns:
        A concise paragraph (3–5 sentences) capturing the key facts,
        preferences, and topics from the session.  Empty string if the
        conversation is too short or if the LLM returns only whitespace.
    """
    if _count_user_turns(conversation_history) < min_turns:
        return ""

    conv_text = _format_conversation(conversation_history)
    if not conv_text.strip():
        return ""

    # Truncate very long transcripts from the front (oldest turns first)
    if len(conv_text) > MAX_CONV_CHARS:
        conv_text = conv_text[-MAX_CONV_CHARS:]
        # Trim to the next newline so we don't start mid-sentence
        newline_idx = conv_text.find("\n")
        if newline_idx != -1:
            conv_text = conv_text[newline_idx + 1:]

    from src.llm.client import OllamaClient
    from src.llm.prompts import SUMMARISE_SESSION_PROMPT

    prompt = SUMMARISE_SESSION_PROMPT.replace("{conversation}", conv_text)
    client = OllamaClient(model=model, base_url=base_url)
    summary = client.chat(
        [{"role": "user", "content": prompt}],
        stream=False,
    ).strip()

    return summary
