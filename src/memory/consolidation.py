"""
Memory consolidation engine — Phase 2.5.

Runs periodically (every CONSOLIDATION_INTERVAL_HOURS) to batch recent session
summaries through the LLM and extract durable claims.  Consolidation state
(last_run_at) is persisted in claims.db so the interval survives restarts.

Lifecycle
---------
  run_if_due()  → checks scheduler → pulls episodes → consolidate() → writes claims
  run_async()   → spawns a daemon thread that calls run_if_due()

Contradiction detection
-----------------------
  Before inserting a new claim, similar existing claims are found with semantic
  search.  If a highly-similar pair is found, a second LLM call classifies the
  relationship as 'agree', 'conflict', or 'independent'.

  - agree    → reinforce the existing claim, skip the new one
  - conflict → lower-confidence claim is marked contradicted_by the higher one
  - independent → both claims are inserted
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.claims import ClaimsStore
    from src.memory.vector_store import MemoryStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSOLIDATION_INTERVAL_HOURS: int = 6
MIN_EPISODES_TO_CONSOLIDATE: int = 3

# Cosine similarity threshold for triggering contradiction check
_CONTRADICTION_SIM_THRESHOLD: float = 0.85

# ---------------------------------------------------------------------------
# Lazy logger
# ---------------------------------------------------------------------------

_log = None


def _get_log():
    global _log
    if _log is None:
        from src.utils.log import get_logger
        _log = get_logger("memory")
    return _log


# ---------------------------------------------------------------------------
# ConsolidationEngine
# ---------------------------------------------------------------------------


class ConsolidationEngine:
    """Extract trust-calibrated claims from episode summaries."""

    def __init__(
        self,
        claims_store: "ClaimsStore",
        memory_store: Optional["MemoryStore"] = None,
        model: str = "phi4-mini",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._claims = claims_store
        self._memory = memory_store
        self.model = model
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(
        self,
        episode_texts: list[str],
        source_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Extract claims from *episode_texts* and write them to the ClaimsStore.

        Each episode is a plain-text session summary.  The LLM is asked to
        pull out atomic, durable facts; each candidate is then:
          1. Passed through the policy gate (should_store)
          2. Checked for agreement/contradiction with existing similar claims
          3. Written to ClaimsStore (or used to reinforce an existing claim)

        Returns the list of claim dicts that were actually written.
        """
        if not episode_texts:
            return []

        from src.llm.client import OllamaClient
        from src.llm.prompts import CONSOLIDATION_PROMPT
        from src.memory.policy import should_store

        transcript = "\n\n---\n\n".join(episode_texts)
        prompt = CONSOLIDATION_PROMPT.replace("{episodes}", transcript)

        try:
            client = OllamaClient(model=self.model, base_url=self.base_url)
            raw = client.chat(
                [{"role": "user", "content": prompt}], stream=False
            )
        except Exception as exc:
            _get_log().error("consolidation LLM call failed: %s", exc)
            return []

        candidates = self._parse_candidates(raw)
        if not candidates:
            _get_log().info(
                "consolidation: no candidates extracted from %d episodes",
                len(episode_texts),
            )
            return []

        # Build a lightweight existing-text dict for the policy gate
        existing_texts: dict[str, bool] = {}

        written: list[dict] = []
        for candidate in candidates:
            text = str(candidate.get("claim", "")).strip()
            category = str(candidate.get("category", "general"))
            confidence = float(candidate.get("confidence", 0.5))
            valid_from = candidate.get("valid_from") or None

            if not text:
                continue

            ok, reason = should_store(text, existing_texts, confidence)
            if not ok:
                _get_log().info(
                    "consolidation: rejected [%s] '%s...' reason=%s",
                    category,
                    text[:50],
                    reason,
                )
                continue

            action = self._contradiction_check(text, confidence)
            if action == "reinforce":
                _get_log().info(
                    "consolidation: reinforced existing claim '%s...'", text[:50]
                )
                existing_texts[text] = True
                continue
            if action == "conflict":
                _get_log().warning(
                    "consolidation: [CONTRADICTION] '%s...'", text[:50]
                )
                # The contradiction_check method already handled the DB writes.
                existing_texts[text] = True
                continue

            try:
                self._claims.add_claim(
                    text=text,
                    category=category,
                    confidence=confidence,
                    valid_from=valid_from,
                    source_ids=source_ids or [],
                )
                existing_texts[text] = True
                written.append(
                    {"text": text, "category": category, "confidence": confidence}
                )
                _get_log().info(
                    "consolidation: wrote claim [%s] '%s...'", category, text[:50]
                )
            except Exception as exc:
                _get_log().error(
                    "consolidation: failed to write '%s...': %s", text[:40], exc
                )

        _get_log().info(
            "consolidation complete: %d episodes → %d/%d candidates written",
            len(episode_texts),
            len(written),
            len(candidates),
        )
        return written

    def run_if_due(self) -> int:
        """Run consolidation if the interval and episode-count thresholds are met.

        Returns the number of new claims written (0 if skipped).
        """
        if self._memory is None:
            return 0

        if not self._is_due():
            return 0

        try:
            episodes = self._get_all_episodes()
        except Exception as exc:
            _get_log().error("run_if_due: failed to load episodes: %s", exc)
            return 0

        if len(episodes) < MIN_EPISODES_TO_CONSOLIDATE:
            _get_log().info(
                "run_if_due: only %d episodes, need %d — skipping",
                len(episodes),
                MIN_EPISODES_TO_CONSOLIDATE,
            )
            return 0

        texts = [ep["text"] for ep in episodes]
        ids = [ep["id"] for ep in episodes]
        written = self.consolidate(texts, source_ids=ids)
        self._record_run()
        return len(written)

    def run_async(self) -> None:
        """Fire-and-forget: spawn a daemon thread that calls run_if_due()."""
        thread = threading.Thread(target=self.run_if_due, daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_candidates(self, raw: str) -> list[dict]:
        """Extract a JSON claims array from the raw LLM response."""
        from src.memory.soul import _extract_json_patch

        try:
            parsed = _extract_json_patch(raw)
            if parsed and isinstance(parsed.get("claims"), list):
                return parsed["claims"]
        except Exception:
            pass
        return []

    def _contradiction_check(self, new_text: str, new_confidence: float) -> str:
        """Compare *new_text* against the most similar existing claims.

        Returns one of:
          'reinforce'   — new_text agrees with an existing claim; call reinforce
          'conflict'    — new_text contradicts an existing claim; DB updated here
          'independent' — no close match; caller should insert as new claim
        """
        similar = self._claims.query_claims(
            new_text, n_results=3, threshold=_CONTRADICTION_SIM_THRESHOLD
        )
        if not similar:
            return "independent"

        from src.llm.client import OllamaClient
        from src.llm.prompts import CONTRADICTION_CHECK_PROMPT

        for existing in similar:
            prompt = CONTRADICTION_CHECK_PROMPT.replace(
                "{claim_a}", existing["text"]
            ).replace("{claim_b}", new_text)

            try:
                client = OllamaClient(model=self.model, base_url=self.base_url)
                verdict = client.chat(
                    [{"role": "user", "content": prompt}], stream=False
                ).strip().lower()
            except Exception:
                continue

            if "agree" in verdict:
                self._claims.reinforce_claim(existing["id"])
                return "reinforce"

            if "conflict" in verdict:
                # Persist the new claim first so we have its ID
                try:
                    new_id = self._claims.add_claim(
                        new_text, confidence=new_confidence
                    )
                except Exception:
                    new_id = None

                if new_id:
                    if existing["confidence"] < new_confidence:
                        self._claims.contradict_claim(
                            existing["id"],
                            by_claim_id=new_id,
                            detail="superseded by higher-confidence claim",
                        )
                    else:
                        self._claims.contradict_claim(
                            new_id,
                            by_claim_id=existing["id"],
                            detail="lower confidence than existing claim",
                        )
                _get_log().warning(
                    "[CONTRADICTION] '%s...' vs '%s...'",
                    existing["text"][:40],
                    new_text[:40],
                )
                return "conflict"

        return "independent"

    # ------------------------------------------------------------------
    # Scheduler helpers (persisted in claims.db)
    # ------------------------------------------------------------------

    def _ensure_scheduler_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS consolidation_state (
                   last_run_at TEXT
               )"""
        )

    def _is_due(self) -> bool:
        """Return True if enough time has passed since the last consolidation run."""
        try:
            with sqlite3.connect(str(self._claims._path), check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                self._ensure_scheduler_table(conn)
                row = conn.execute(
                    "SELECT last_run_at FROM consolidation_state LIMIT 1"
                ).fetchone()
                if row is None or row["last_run_at"] is None:
                    return True
                cutoff = datetime.utcnow() - timedelta(
                    hours=CONSOLIDATION_INTERVAL_HOURS
                )
                return datetime.fromisoformat(row["last_run_at"]) < cutoff
        except Exception:
            return True

    def _record_run(self) -> None:
        now = datetime.utcnow().isoformat(timespec="seconds")
        try:
            with sqlite3.connect(str(self._claims._path), check_same_thread=False) as conn:
                conn.row_factory = sqlite3.Row
                self._ensure_scheduler_table(conn)
                existing = conn.execute(
                    "SELECT rowid FROM consolidation_state LIMIT 1"
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE consolidation_state SET last_run_at = ?", (now,)
                    )
                else:
                    conn.execute(
                        "INSERT INTO consolidation_state (last_run_at) VALUES (?)",
                        (now,),
                    )
        except Exception as exc:
            _get_log().error("_record_run failed: %s", exc)

    def _get_all_episodes(self) -> list[dict]:
        """Return all documents from MemoryStore as {id, text} dicts."""
        if self._memory is None:
            return []
        return self._memory.get_all_memories()
