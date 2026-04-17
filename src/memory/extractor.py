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
    from src.memory.vector_store import MemoryStore

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
        soul_data: Optional[dict] = None,
        vector_store: Optional["MemoryStore"] = None,
    ) -> None:
        """Run each candidate through ``should_store()`` and persist approved facts.

        Facts passing the ``"long_term"`` gate are written to *vector_store* so
        they can be retrieved via RAG in future sessions.  Facts rated
        ``"short_term_only"`` are logged but not persisted.

        Args:
            candidates:   Output of ``extract_candidates()``.
            soul:         Live ``SoulFile`` instance for repeat-detection.
            soul_data:    Pre-loaded soul dict (avoids a second file read if
                          you already have it; falls back to ``soul.load()``).
            vector_store: ``MemoryStore`` instance to write long-term facts into.
        """
        from src.memory.policy import should_store

        store = soul_data if soul_data is not None else soul.load()

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
            if store_flag and reason == "long_term" and vector_store is not None:
                try:
                    vector_store.add_memory(
                        fact,
                        {
                            "source": "extractor",
                            "category": item.get("category", "unknown"),
                        },
                    )
                    _get_log().info("extractor: persisted to vector store — %r", fact)
                except Exception as exc:
                    _get_log().error("extractor: failed to persist %r: %s", fact, exc)


# ---------------------------------------------------------------------------
# Thread worker
# ---------------------------------------------------------------------------


def _extract_worker(
    extractor: MemoryExtractor,
    soul: "SoulFile",
    conversation_snapshot: list[dict],
    vector_store: Optional["MemoryStore"] = None,
) -> None:
    """Worker executed in a daemon thread — extract facts from recent user turns."""
    log = _get_log()
    log.info("extract_worker start — msgs=%d", len(conversation_snapshot))
    try:
        # Scan the last 10 messages so facts stated a few turns ago are not missed.
        # The last user turn alone is too narrow — by the time EXTRACT_EVERY fires,
        # the informative turns may already be behind the cursor.
        recent = conversation_snapshot[-10:]
        user_turns = [m for m in recent if m.get("role") == "user"]
        if not user_turns:
            log.info("extract_worker: no user turns found")
            return

        all_candidates: list[dict] = []
        seen_facts: set[str] = set()
        for turn in user_turns:
            for c in extractor.extract_candidates(turn):
                key = c["fact"].lower().strip()
                if key not in seen_facts:
                    seen_facts.add(key)
                    all_candidates.append(c)

        if not all_candidates:
            log.info("extract_worker: no candidates found")
            return

        log.info("extract_worker: %d candidate(s) found across %d turns",
                 len(all_candidates), len(user_turns))
        extractor.commit(all_candidates, soul, vector_store=vector_store)

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
    vector_store: Optional["MemoryStore"] = None,
) -> None:
    """Spawn a daemon thread to extract and evaluate memory candidates.

    Mirror of ``maybe_update_soul()`` — fire-and-forget, never blocks the
    main response path.

    Args:
        soul:         Live ``SoulFile`` instance.
        conversation: Current conversation history (will be snapshot-copied).
        model:        Ollama model tag to use for extraction.
        base_url:     Ollama server URL.
        vector_store: ``MemoryStore`` instance for persisting long-term facts.
    """
    extractor = MemoryExtractor(model=model, base_url=base_url)
    thread = threading.Thread(
        target=_extract_worker,
        args=(extractor, soul, list(conversation), vector_store),
        daemon=True,
    )
    thread.start()
