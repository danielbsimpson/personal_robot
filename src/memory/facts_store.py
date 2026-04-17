"""
Structured facts store — Orion's secondary, retrieve-on-demand knowledge base.

Stores durable facts that are too detailed for the always-on soul file but too
structured for the unstructured vector store.  Facts are grouped into categories
and surface in the system prompt only when the user's message is relevant to
them.

Layout
------
``data/facts.json`` is a flat JSON file with the shape::

    {
        "<category>": [
            {
                "fact": "Daniel is a Data Scientist Manager at TJX Companies.",
                "confidence": 1.0,
                "explicit": true,
                "source": "migration",
                "ts": "2026-04-17T12:00:00"
            },
            ...
        ],
        ...
    }

Categories
----------
``work``            Job titles, employer, department, focus areas, tools, projects
``education``       Degrees, institutions, dissertation topics
``interests``       Hobbies, sports, music, games, entertainment
``skills``          Technical skills, languages, platforms
``relationships``   Extended family, friends, named connections
``partner``         Partner's career, education, projects, exhibitions
``travel``          Lived-in places, countries visited
``projects``        Personal and open-source projects
``general``         Anything that does not fit elsewhere

Retrieval
---------
``query_facts(message)`` performs two passes:

1. **Category pass** — maps trigger keywords in the message to one or more
   categories and returns all facts in those categories.
2. **Keyword fallback** — scans all remaining facts for any word from the
   message that appears in the fact text (case-insensitive).

Results are deduplicated and returned as plain strings ready for injection.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FACTS_PATH = _PROJECT_ROOT / "data" / "facts.json"

# ---------------------------------------------------------------------------
# Category keyword triggers
# Mapping: category → set of trigger words (lower-case)
# ---------------------------------------------------------------------------

_CATEGORY_TRIGGERS: dict[str, frozenset[str]] = {
    "work": frozenset(
        {
            "job", "work", "career", "company", "employer", "boss", "manager",
            "profession", "role", "title", "department", "office", "colleague",
            "tjx", "data scientist", "marketing", "analytics", "pipeline",
            "model", "dashboard", "snowflake", "databricks",
        }
    ),
    "education": frozenset(
        {
            "degree", "university", "college", "school", "study", "studied",
            "graduate", "graduation", "dissertation", "thesis", "phd", "msc",
            "bsc", "masters", "bachelor", "academic",
        }
    ),
    "interests": frozenset(
        {
            "hobby", "hobbies", "interest", "like", "love", "enjoy", "favourite",
            "music", "sport", "game", "film", "book", "reading", "cook", "cooking",
            "travel", "exercise", "gym", "climbing", "hiking", "martial arts",
            "muay thai", "boxing", "mma", "photography", "d&d", "dungeons",
            "dragons", "video game", "sci-fi", "fantasy",
        }
    ),
    "skills": frozenset(
        {
            "skill", "skills", "language", "python", "sql", "r ", "javascript",
            "html", "css", "streamlit", "tensorflow", "keras", "scikit",
            "pandas", "plotly", "dash", "pyspark", "ml", "machine learning",
            "deep learning", "nlp", "computer vision", "llm",
        }
    ),
    "relationships": frozenset(
        {
            "family", "mum", "mom", "mother", "dad", "father", "brother",
            "sister", "parent", "sibling", "step", "relative", "friend",
        }
    ),
    "partner": frozenset(
        {
            "danielle", "dani", "wife", "partner", "studio", "painting",
            "art", "artist", "exhibition", "phd", "goldsmiths",
        }
    ),
    "travel": frozenset(
        {
            "travel", "trip", "visit", "country", "lived", "thailand",
            "london", "england", "virginia", "boston", "framingham",
            "west virginia",
        }
    ),
    "projects": frozenset(
        {
            "project", "github", "portfolio", "app", "dashboard", "build",
            "personal project", "side project", "kaggle", "open source",
        }
    ),
}

# Minimum number of characters in a message word to count as a keyword signal
_MIN_KEYWORD_LEN: int = 4

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
# FactsStore
# ---------------------------------------------------------------------------


class FactsStore:
    """Read/write wrapper for ``data/facts.json``.

    Thread-safe: all writes are serialised through a module-level lock.
    """

    def __init__(self, path: Path | str = DEFAULT_FACTS_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core IO
    # ------------------------------------------------------------------

    def load(self) -> dict[str, list[dict]]:
        """Return the full facts dict.  Returns empty dict if file is absent."""
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
            except (json.JSONDecodeError, ValueError):
                return {}

    def save(self, data: dict[str, list[dict]]) -> None:
        """Write *data* atomically via a temp file."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp.replace(self.path)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        fact: str,
        category: str = "general",
        confidence: float = 1.0,
        explicit: bool = True,
        source: str = "auto",
    ) -> bool:
        """Add *fact* to *category* if it is not already present.

        Returns True if the fact was added, False if it was a duplicate.
        """
        fact = fact.strip()
        if not fact:
            return False

        data = self.load()
        existing = data.get(category, [])

        # Deduplicate: skip if any existing entry contains substantially the
        # same text (case-insensitive substring match).
        fact_lower = fact.lower()
        for entry in existing:
            if fact_lower in entry.get("fact", "").lower():
                return False

        entry = {
            "fact": fact,
            "confidence": confidence,
            "explicit": explicit,
            "source": source,
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
        }
        data.setdefault(category, []).append(entry)
        self.save(data)
        _get_log().info("facts_store: added [%s] %r", category, fact[:80])
        return True

    def add_facts_bulk(self, entries: list[dict]) -> int:
        """Add multiple facts at once.

        Each entry should have keys: ``fact``, ``category``, optionally
        ``confidence``, ``explicit``, ``source``.

        Returns the number of facts actually added (duplicates skipped).
        """
        data = self.load()
        added = 0
        for e in entries:
            fact = str(e.get("fact", "")).strip()
            if not fact:
                continue
            category = str(e.get("category", "general"))
            confidence = float(e.get("confidence", 1.0))
            explicit = bool(e.get("explicit", True))
            source = str(e.get("source", "auto"))

            fact_lower = fact.lower()
            existing = data.get(category, [])
            if any(fact_lower in entry.get("fact", "").lower() for entry in existing):
                continue

            data.setdefault(category, []).append(
                {
                    "fact": fact,
                    "confidence": confidence,
                    "explicit": explicit,
                    "source": source,
                    "ts": datetime.utcnow().isoformat(timespec="seconds"),
                }
            )
            added += 1

        if added:
            self.save(data)
            _get_log().info("facts_store: bulk-added %d facts", added)
        return added

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def query_facts(self, message: str, max_facts: int = 20) -> list[str]:
        """Return a list of fact strings relevant to *message*.

        Two-pass retrieval:
        1. Category pass — keyword triggers map the message to categories.
        2. Keyword fallback — scans remaining facts for word overlap.

        Results are deduplicated and capped at *max_facts*.
        """
        if not message or not message.strip():
            return []

        data = self.load()
        if not data:
            return []

        message_lower = message.lower()
        message_words = frozenset(
            w for w in re.split(r"\W+", message_lower) if len(w) >= _MIN_KEYWORD_LEN
        )

        # --- Pass 1: category triggers ---
        matched_categories: set[str] = set()
        for category, triggers in _CATEGORY_TRIGGERS.items():
            if category not in data:
                continue
            for trigger in triggers:
                if trigger in message_lower:
                    matched_categories.add(category)
                    break

        results: list[str] = []
        seen: set[str] = set()

        for category in matched_categories:
            for entry in data.get(category, []):
                fact = entry.get("fact", "")
                if fact and fact not in seen:
                    results.append(fact)
                    seen.add(fact)

        # --- Pass 2: keyword fallback across remaining categories ---
        for category, entries in data.items():
            if category in matched_categories:
                continue
            for entry in entries:
                fact = entry.get("fact", "")
                if not fact or fact in seen:
                    continue
                fact_words = frozenset(
                    w for w in re.split(r"\W+", fact.lower()) if len(w) >= _MIN_KEYWORD_LEN
                )
                if fact_words & message_words:
                    results.append(fact)
                    seen.add(fact)

        if results:
            _get_log().info(
                "facts_store: query returned %d facts for message (len=%d)",
                len(results),
                len(message),
            )

        return results[:max_facts]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def all_facts_flat(self) -> list[str]:
        """Return every fact as a plain string, regardless of category."""
        data = self.load()
        return [
            entry.get("fact", "")
            for entries in data.values()
            for entry in entries
            if entry.get("fact")
        ]

    def category_count(self) -> dict[str, int]:
        """Return {category: count} for the current facts file."""
        return {cat: len(entries) for cat, entries in self.load().items()}
