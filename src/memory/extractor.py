"""
Two-pass memory extraction — runs as a daemon thread after each turn (configurable
cadence) to decide what new facts are worth committing to the soul file or the
long-term vector store.

Keeps extraction entirely separate from the main conversation client so it never
blocks the response path.

Usage
-----
From app.py / main.py, call ``maybe_extract_memories()`` the same way you call
``maybe_update_soul()`` — it is a fire-and-forget daemon thread.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.memory.soul import SoulFile

# How many user turns between extraction passes.
# Deliberately offset from SOUL_UPDATE_EVERY (3) to avoid piling LLM calls.
EXTRACT_EVERY: int = 5

# ---------------------------------------------------------------------------
# Module-level lazy logger
# ---------------------------------------------------------------------------

_log = None


def _get_log():
    global _log
    if _log is None:
        from src.utils.log import get_logger
        _log = get_logger("memory")
    return _log


# ---------------------------------------------------------------------------
# MemoryExtractor class
# ---------------------------------------------------------------------------


class MemoryExtractor:
    """Extracts factual candidates from a single conversation turn via LLM.

    Uses its own OllamaClient instance so extraction never shares state with
    the main chat client.  Results are evaluated through ``should_store()`` and
    logged to ``data/logs/memory.log`` for Phase 2.4 wiring.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url

    def extract_candidates(self, turn: dict) -> list[dict]:
        """Ask the LLM to extract factual claims from a single user turn.

        Args:
            turn: A ``{"role": ..., "content": ...}`` message dict.

        Returns:
            List of ``{fact, category, confidence, explicit}`` dicts.
            Empty list on failure or when nothing is found.
        """
        if turn.get("role") != "user" or not turn.get("content", "").strip():
            return []

        from src.llm.client import OllamaClient
        from src.llm.prompts import MEMORY_EXTRACT_PROMPT
        from src.memory.soul import _extract_json_patch

        prompt = MEMORY_EXTRACT_PROMPT.format(message=turn["content"])
        try:
            client = OllamaClient(model=self.model, base_url=self.base_url)
            raw = client.chat([{"role": "user", "content": prompt}], stream=False)
            parsed = _extract_json_patch(raw)
            if not parsed:
                return []

            candidates = parsed.get("candidates", [])
            if not isinstance(candidates, list):
                return []

            valid: list[dict] = []
            for c in candidates:
                if isinstance(c, dict) and "fact" in c and "confidence" in c:
                    valid.append(
                        {
                            "fact": str(c["fact"]),
                            "category": str(c.get("category", "unknown")),
                            "confidence": float(c.get("confidence", 0.0)),
                            "explicit": bool(c.get("explicit", False)),
                        }
                    )
            return valid

        except Exception as exc:
            _get_log().error("extract_candidates error: %s", exc, exc_info=True)
            return []

    def commit(
        self,
        candidates: list[dict],
        soul: "SoulFile",
        memory_store: Optional[dict] = None,
    ) -> None:
        """Run each candidate through ``should_store()`` and log the decision.

        Phase 2.4 will wire the approved candidates into ``apply_patch()`` /
        ``MemoryStore.add_memory()``.  For now this method only logs, keeping
        the extractor safe to enable before the full RAG stack is in place.

        Args:
            candidates:    Output of ``extract_candidates()``.
            soul:          Live ``SoulFile`` instance for repeat-detection.
            memory_store:  Pre-loaded soul dict (avoids a second file read if
                           you already have it; falls back to ``soul.load()``).
        """
        from src.memory.policy import should_store

        store = memory_store if memory_store is not None else soul.load()

        for item in candidates:
            fact = item["fact"]
            confidence = item["confidence"]
            store_flag, reason = should_store(fact, store, confidence)
            _get_log().info(
                "extractor: candidate=%r category=%s conf=%.2f explicit=%s "
                "→ store=%s reason=%s",
                fact,
                item.get("category"),
                confidence,
                item.get("explicit"),
                store_flag,
                reason,
            )


# ---------------------------------------------------------------------------
# Thread worker
# ---------------------------------------------------------------------------


def _extract_worker(
    extractor: MemoryExtractor,
    soul: "SoulFile",
    conversation_snapshot: list[dict],
) -> None:
    """Worker executed in a daemon thread — extract and evaluate the last user turn."""
    log = _get_log()
    log.info("extract_worker start — msgs=%d", len(conversation_snapshot))
    try:
        user_turns = [m for m in conversation_snapshot if m.get("role") == "user"]
        if not user_turns:
            log.info("extract_worker: no user turns found")
            return

        last_turn = user_turns[-1]
        candidates = extractor.extract_candidates(last_turn)
        if not candidates:
            log.info("extract_worker: no candidates found")
            return

        log.info("extract_worker: %d candidate(s) found", len(candidates))
        extractor.commit(candidates, soul)

    except Exception as exc:
        log.error("extract_worker error: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def maybe_extract_memories(
    soul: "SoulFile",
    conversation: list[dict],
    model: str,
    base_url: str = "http://localhost:11434",
) -> None:
    """Spawn a daemon thread to extract and evaluate memory candidates.

    Mirror of ``maybe_update_soul()`` — fire-and-forget, never blocks the
    main response path.

    Args:
        soul:         Live ``SoulFile`` instance.
        conversation: Current conversation history (will be snapshot-copied).
        model:        Ollama model tag to use for extraction.
        base_url:     Ollama server URL.
    """
    extractor = MemoryExtractor(model=model, base_url=base_url)
    thread = threading.Thread(
        target=_extract_worker,
        args=(extractor, soul, list(conversation)),
        daemon=True,
    )
    thread.start()
