"""
Claims store — trust-calibrated, deduplicated long-term memory for Orion.

Sits above the raw episode store (ChromaDB session summaries) and below
the soul file.  Claims are LLM-extracted, atomic, single-sentence facts with:
  - provenance (source episode IDs)
  - temporal validity (valid_from, valid_until)
  - trust signals (confidence, reinforcement_count, contradicted_by)
  - full event audit trail (claim_events table)
  - embedding stored in-row for fast similarity scan

Retrieval pipeline:
  1. Semantic vector scan (numpy cosine over stored float embeddings)
  2. SQLite FTS5 keyword fallback when no vector results clear threshold
  3. Trust score sort: confidence × (1 + log(1 + reinforcement_count)) × recency

Trust score formula:
    recency = exp(-days_since_last_reinforcement / DECAY_DAYS)
    trust   = confidence × (1 + log(1 + reinforcement_count)) × recency

A brand-new claim (rc=0, age=0 days) scores: 1.0 × 1.0 × 1.0 = 1.0
After 5 reinforcements:                       1.0 × (1+ln6) × recency
After 30 days unreinforced:                   1.0 × 1.0 × e⁻¹ ≈ 0.37
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CLAIMS_DB = _PROJECT_ROOT / "data" / "memory" / "claims.db"

DECAY_DAYS: int = 30
VECTOR_THRESHOLD: float = 0.40
_MIN_CLAIM_LEN: int = 10
_SCHEMA_VERSION: int = 1

# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id                   TEXT PRIMARY KEY,
    text                 TEXT NOT NULL,
    category             TEXT NOT NULL DEFAULT 'general',
    confidence           REAL NOT NULL DEFAULT 1.0,
    valid_from           TEXT NOT NULL,
    valid_until          TEXT,
    source_episode_ids   TEXT NOT NULL DEFAULT '[]',
    reinforcement_count  INTEGER NOT NULL DEFAULT 0,
    last_reinforced_at   TEXT NOT NULL,
    contradicted_by      TEXT,
    created_at           TEXT NOT NULL,
    embedding            TEXT
);

CREATE TABLE IF NOT EXISTS claim_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id   TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '{}',
    ts         TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
    text,
    content=claims,
    content_rowid=rowid
);

CREATE TRIGGER IF NOT EXISTS claims_fts_ins AFTER INSERT ON claims BEGIN
    INSERT INTO claims_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS claims_fts_upd AFTER UPDATE OF text ON claims BEGIN
    INSERT INTO claims_fts(claims_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO claims_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS claims_fts_del AFTER DELETE ON claims BEGIN
    INSERT INTO claims_fts(claims_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
"""

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _claim_id(text: str) -> str:
    """Deterministic 32-char ID: SHA-256 of lower-cased whitespace-normalised text."""
    normalised = " ".join(text.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:32]


def _trust_score(
    confidence: float,
    reinforcement_count: int,
    last_reinforced_at: str,
) -> float:
    """Compute a decayed trust score for ranking retrieved claims.

    Higher is more trustworthy.  New claims start at ``confidence``.
    Each reinforcement multiplies by (1 + log(1 + count)).
    Recency decays exponentially with a half-life of DECAY_DAYS.
    """
    try:
        last = datetime.fromisoformat(last_reinforced_at)
    except (ValueError, TypeError):
        last = datetime.utcnow()
    days_old = max(0.0, (datetime.utcnow() - last).total_seconds() / 86400)
    recency = math.exp(-days_old / DECAY_DAYS)
    return confidence * (1.0 + math.log1p(reinforcement_count)) * recency


# ---------------------------------------------------------------------------
# ClaimsStore
# ---------------------------------------------------------------------------


class ClaimsStore:
    """SQLite-backed trust-calibrated claims store."""

    def __init__(self, db_path: Path | str = DEFAULT_CLAIMS_DB) -> None:
        self._path = Path(db_path)
        self._lock = threading.Lock()
        self._embedder = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema bootstrapping
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            row = conn.execute(
                "SELECT version FROM schema_version LIMIT 1"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_version VALUES (?)", (_SCHEMA_VERSION,)
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _get_embedder(self):
        if self._embedder is None:
            from src.memory.embeddings import Embedder
            self._embedder = Embedder()
        return self._embedder

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add_claim(
        self,
        text: str,
        category: str = "general",
        confidence: float = 1.0,
        valid_from: Optional[str] = None,
        source_ids: Optional[list[str]] = None,
    ) -> str:
        """Insert or reinforce an existing claim.

        Deduplication is by SHA-256 of the normalised claim text so re-adding
        the same fact merely increments its reinforcement count.

        Returns the claim ID (stable across calls for the same text).
        """
        text = text.strip()
        if not text or len(text) < _MIN_CLAIM_LEN:
            raise ValueError(f"Claim text too short or empty: {text!r}")

        claim_id = _claim_id(text)
        now = _now_iso()
        valid_from = valid_from or now
        source_ids_json = json.dumps(source_ids or [])

        # Compute embedding eagerly so it is stored alongside the claim.
        try:
            embedding_json = json.dumps(self._get_embedder().embed(text))
        except Exception:
            embedding_json = None

        with self._lock:
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT id, reinforcement_count, source_episode_ids "
                    "FROM claims WHERE id = ?",
                    (claim_id,),
                ).fetchone()

                if existing:
                    old_sources = json.loads(existing["source_episode_ids"] or "[]")
                    merged = list(dict.fromkeys(old_sources + (source_ids or [])))
                    conn.execute(
                        """UPDATE claims
                           SET reinforcement_count  = reinforcement_count + 1,
                               last_reinforced_at   = ?,
                               source_episode_ids   = ?,
                               confidence           = MAX(confidence, ?)
                           WHERE id = ?""",
                        (now, json.dumps(merged), confidence, claim_id),
                    )
                    conn.execute(
                        "INSERT INTO claim_events "
                        "(claim_id, event_type, detail, ts) VALUES (?, ?, ?, ?)",
                        (
                            claim_id,
                            "reinforced",
                            json.dumps(
                                {"count": existing["reinforcement_count"] + 1}
                            ),
                            now,
                        ),
                    )
                else:
                    conn.execute(
                        """INSERT INTO claims
                           (id, text, category, confidence, valid_from, valid_until,
                            source_episode_ids, reinforcement_count, last_reinforced_at,
                            contradicted_by, created_at, embedding)
                           VALUES (?, ?, ?, ?, ?, NULL, ?, 0, ?, NULL, ?, ?)""",
                        (
                            claim_id,
                            text,
                            category,
                            confidence,
                            valid_from,
                            source_ids_json,
                            now,
                            now,
                            embedding_json,
                        ),
                    )
                    conn.execute(
                        "INSERT INTO claim_events "
                        "(claim_id, event_type, detail, ts) VALUES (?, ?, ?, ?)",
                        (
                            claim_id,
                            "created",
                            json.dumps(
                                {"category": category, "confidence": confidence}
                            ),
                            now,
                        ),
                    )

        return claim_id

    def reinforce_claim(self, claim_id: str) -> None:
        """Increment reinforcement_count and refresh last_reinforced_at."""
        now = _now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """UPDATE claims
                       SET reinforcement_count = reinforcement_count + 1,
                           last_reinforced_at  = ?
                       WHERE id = ?""",
                    (now, claim_id),
                )
                conn.execute(
                    "INSERT INTO claim_events "
                    "(claim_id, event_type, detail, ts) VALUES (?, ?, ?, ?)",
                    (claim_id, "reinforced", "{}", now),
                )

    def contradict_claim(
        self, claim_id: str, by_claim_id: str, detail: str = ""
    ) -> None:
        """Mark *claim_id* as contradicted by *by_claim_id*.

        Does not delete the claim — it is kept in the audit trail but excluded
        from normal query results.
        """
        now = _now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE claims SET contradicted_by = ? WHERE id = ?",
                    (by_claim_id, claim_id),
                )
                conn.execute(
                    "INSERT INTO claim_events "
                    "(claim_id, event_type, detail, ts) VALUES (?, ?, ?, ?)",
                    (
                        claim_id,
                        "contradicted",
                        json.dumps({"by": by_claim_id, "detail": detail}),
                        now,
                    ),
                )

    def expire_claim(self, claim_id: str) -> None:
        """Set valid_until = now() so the claim is excluded from future queries."""
        now = _now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE claims SET valid_until = ? WHERE id = ?",
                    (now, claim_id),
                )
                conn.execute(
                    "INSERT INTO claim_events "
                    "(claim_id, event_type, detail, ts) VALUES (?, ?, ?, ?)",
                    (claim_id, "expired", "{}", now),
                )

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def query_claims(
        self,
        text: str,
        n_results: int = 8,
        threshold: float = VECTOR_THRESHOLD,
        as_of: Optional[str] = None,
        include_contradicted: bool = False,
    ) -> list[dict]:
        """Return trust-ranked claims semantically similar to *text*.

        Steps:
          1. Load all active claims from SQLite.
          2. Compute cosine similarity of *text* embedding against each stored embedding.
          3. Filter to those meeting *threshold*.
          4. Fall back to FTS5 keyword search if vector pass finds nothing.
          5. Sort by trust score descending, return top *n_results*.

        Temporal filter: claims where ``valid_until < as_of`` are excluded.
        """
        if not text or not text.strip():
            return []

        now_str = as_of or _now_iso()
        query_vec = None

        try:
            import numpy as np
            embedder = self._get_embedder()
            q = np.array(embedder.embed(text), dtype=float)
            q_norm = np.linalg.norm(q)
            if q_norm > 0:
                query_vec = q / q_norm
        except Exception:
            pass

        matched: list[dict] = []

        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM claims
                   WHERE (valid_until IS NULL OR valid_until > ?)
                     AND (? OR contradicted_by IS NULL)""",
                (now_str, include_contradicted),
            ).fetchall()

            if query_vec is not None:
                import numpy as np
                for row in rows:
                    emb_raw = row["embedding"]
                    if not emb_raw:
                        continue
                    try:
                        vec = np.array(json.loads(emb_raw), dtype=float)
                        v_norm = np.linalg.norm(vec)
                        if v_norm == 0:
                            continue
                        sim = float(np.dot(query_vec, vec / v_norm))
                    except Exception:
                        continue
                    if sim >= threshold:
                        d = dict(row)
                        d["_sim"] = sim
                        d["_trust"] = _trust_score(
                            d["confidence"],
                            d["reinforcement_count"],
                            d["last_reinforced_at"],
                        )
                        matched.append(d)

            # FTS5 fallback: triggered when vector search returns nothing
            if not matched:
                try:
                    fts_rows = conn.execute(
                        """SELECT c.* FROM claims c
                           JOIN claims_fts f ON c.rowid = f.rowid
                           WHERE claims_fts MATCH ?
                             AND (c.valid_until IS NULL OR c.valid_until > ?)
                             AND (? OR c.contradicted_by IS NULL)
                           ORDER BY rank
                           LIMIT ?""",
                        (text, now_str, include_contradicted, n_results),
                    ).fetchall()
                    for row in fts_rows:
                        d = dict(row)
                        d["_sim"] = 0.0
                        d["_trust"] = _trust_score(
                            d["confidence"],
                            d["reinforcement_count"],
                            d["last_reinforced_at"],
                        )
                        matched.append(d)
                except Exception:
                    # FTS query syntax errors (e.g. special chars) are non-fatal
                    pass

        matched.sort(key=lambda x: x["_trust"], reverse=True)
        return matched[:n_results]

    def decay_report(self, threshold_days: int = DECAY_DAYS) -> list[dict]:
        """Return active claims not reinforced within *threshold_days* days."""
        cutoff = (
            datetime.utcnow() - timedelta(days=threshold_days)
        ).isoformat(timespec="seconds")
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM claims
                   WHERE last_reinforced_at < ?
                     AND valid_until IS NULL
                   ORDER BY last_reinforced_at ASC""",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def timeline(
        self,
        from_dt: Optional[str] = None,
        to_dt: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return claim_events joined with claim text, newest first."""
        from_dt = from_dt or "1970-01-01"
        to_dt = to_dt or _now_iso()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT e.id, e.claim_id, c.text AS claim_text,
                          e.event_type, e.detail, e.ts
                   FROM claim_events e
                   LEFT JOIN claims c ON c.id = e.claim_id
                   WHERE e.ts BETWEEN ? AND ?
                   ORDER BY e.ts DESC
                   LIMIT ?""",
                (from_dt, to_dt, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> dict:
        """Return summary counts: total active, stale, contradicted."""
        cutoff = (
            datetime.utcnow() - timedelta(days=DECAY_DAYS)
        ).isoformat(timespec="seconds")
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM claims WHERE valid_until IS NULL"
            ).fetchone()[0]
            contradicted = conn.execute(
                "SELECT COUNT(*) FROM claims "
                "WHERE contradicted_by IS NOT NULL AND valid_until IS NULL"
            ).fetchone()[0]
            stale = conn.execute(
                "SELECT COUNT(*) FROM claims "
                "WHERE last_reinforced_at < ? AND valid_until IS NULL",
                (cutoff,),
            ).fetchone()[0]
        return {"total": total, "contradicted": contradicted, "stale": stale}

    def import_facts(self, facts: list[dict]) -> int:
        """Bulk-import fact dicts as claims, skipping duplicates.

        Each dict must have a ``fact`` key (str).  Optional: ``category``,
        ``confidence``.  Returns the count of *newly inserted* claims.
        """
        added = 0
        for f in facts:
            text = str(f.get("fact", "")).strip()
            if not text or len(text) < _MIN_CLAIM_LEN:
                continue
            cid = _claim_id(text)
            with self._connect() as conn:
                exists = conn.execute(
                    "SELECT id FROM claims WHERE id = ?", (cid,)
                ).fetchone()
            if exists:
                continue
            try:
                self.add_claim(
                    text=text,
                    category=str(f.get("category", "general")),
                    confidence=float(f.get("confidence", 1.0)),
                    source_ids=["import"],
                )
                added += 1
            except Exception:
                pass
        return added
