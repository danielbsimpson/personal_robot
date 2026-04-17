"""
Soul file — Orion's persistent identity and long-term facts.

Manages data/soul.yaml: a structured YAML record of who the robot is,
who the user is, and what the robot has learned about the world.

After every SOUL_UPDATE_EVERY user turns, the LLM is asked (in a background
daemon thread) whether it learned any new facts worth keeping. If it responds
with a JSON patch block the patch is applied to the soul file automatically.
"""

import json
import logging
import re
import threading
from pathlib import Path
from typing import Optional

import yaml

from src.utils.log import get_logger

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# Resolve project root from this file: src/memory/soul.py → ../../..
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SOUL_PATH = _PROJECT_ROOT / "data" / "soul.yaml"

# How many user messages between background patch checks
SOUL_UPDATE_EVERY = 3

# How many user messages between curiosity-question generation passes
SOUL_CURIOSITY_EVERY = 6

# Module-level lock prevents concurrent writes from multiple threads
_write_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Lazy logger — writes to data/logs/soul_changes.log via get_logger
# ---------------------------------------------------------------------------

_log: Optional[logging.Logger] = None


def _get_log() -> logging.Logger:
    """Return the soul_changes logger, creating it on first call."""
    global _log
    if _log is None:
        _log = get_logger("soul_changes")
    return _log


# ---------------------------------------------------------------------------
# SoulFile class
# ---------------------------------------------------------------------------

class SoulFile:
    """Read/write wrapper for the soul YAML file."""

    def __init__(self, path: Path | str = DEFAULT_SOUL_PATH) -> None:
        self.path = Path(path)

    def load(self) -> dict:
        """Parse and return the soul YAML. Returns empty dict if file missing."""
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def save(self, data: dict) -> None:
        """Write data to the soul file atomically via temp-file + rename."""
        with _write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".yaml.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            tmp.replace(self.path)

    def to_prompt_section(self, budget_chars: Optional[int] = None) -> str:
        """Return a formatted ## About Me YAML block for system prompt injection.

        If *budget_chars* is set and the full YAML exceeds it, sections are
        progressively dropped in this priority order (least important first):
          1. ``facts``
          2. ``environment``
          3. Extended ``user`` fields — keeps only name, preferred_name,
             date_of_birth, location; drops everything else
          4. ``identity`` non-essentials — keeps name, persona,
             communication_style; drops capabilities, hardware, curiosity_queue

        Each drop level is tried in sequence until the text fits.
        """
        data = self.load()
        if not data:
            return ""

        def _render(d: dict) -> str:
            soul_yaml = yaml.dump(
                d, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
            return f"## About Me\n\n```yaml\n{soul_yaml}```"

        text = _render(data)
        if budget_chars is None or len(text) <= budget_chars:
            return text

        # --- Level 1: drop facts ---
        trimmed = {k: v for k, v in data.items() if k != "facts"}
        text = _render(trimmed)
        if len(text) <= budget_chars:
            _get_log().info("soul trimmed: dropped 'facts' section")
            return text

        # --- Level 2: drop environment ---
        trimmed = {k: v for k, v in trimmed.items() if k != "environment"}
        text = _render(trimmed)
        if len(text) <= budget_chars:
            _get_log().info("soul trimmed: dropped 'facts' and 'environment' sections")
            return text

        # --- Level 3: keep only core user fields ---
        _USER_CORE = {"name", "preferred_name", "date_of_birth", "location"}
        if "user" in trimmed and isinstance(trimmed["user"], dict):
            trimmed["user"] = {
                k: v for k, v in trimmed["user"].items() if k in _USER_CORE
            }
        text = _render(trimmed)
        if len(text) <= budget_chars:
            _get_log().info("soul trimmed: reduced user to core fields")
            return text

        # --- Level 4: keep only core identity fields ---
        _IDENTITY_CORE = {"name", "persona", "communication_style"}
        if "identity" in trimmed and isinstance(trimmed["identity"], dict):
            trimmed["identity"] = {
                k: v for k, v in trimmed["identity"].items() if k in _IDENTITY_CORE
            }
        text = _render(trimmed)
        if len(text) <= budget_chars:
            _get_log().info("soul trimmed: reduced identity to core fields")
            return text

        # --- Level 5: trim partner to core fields ---
        _PARTNER_CORE = {"name", "preferred_name", "relationship"}
        if "partner" in trimmed and isinstance(trimmed["partner"], dict):
            trimmed["partner"] = {
                k: v for k, v in trimmed["partner"].items() if k in _PARTNER_CORE
            }
        text = _render(trimmed)
        if len(text) <= budget_chars:
            _get_log().info("soul trimmed: reduced partner to core fields")
        else:
            _get_log().warning("soul still over budget after maximum trimming (%d chars)", len(text))
        return text

    def update(self, section: str, key: str, value: object) -> None:
        """Set a single key within a section and save."""
        data = self.load()
        if section not in data or not isinstance(data.get(section), dict):
            data[section] = {}
        data[section][key] = value
        self.save(data)

    def apply_patch(self, patch: dict) -> None:
        """Merge a patch dict shaped {section: {key: value}} into the soul file."""
        if not patch:
            return
        data = self.load()

        # Record before/after for each changed key for the audit log
        changes: dict[str, dict] = {}
        for section, updates in patch.items():
            if isinstance(updates, dict):
                for key, new_val in updates.items():
                    # Skip empty-dict values — the model sometimes returns
                    # {"family": {}} when it detects family content but extracts
                    # nothing concrete.  Writing these would be a no-op at best
                    # and misleading in the audit log.
                    if isinstance(new_val, dict) and not new_val:
                        continue
                    old_val = (data.get(section) or {}).get(key, "<new>")
                    changes[f"{section}.{key}"] = {"before": old_val, "after": new_val}
            else:
                changes[section] = {"before": data.get(section, "<new>"), "after": updates}

        if not changes:
            _get_log().info("apply_patch: all values were empty — skipping write")
            return

        for section, updates in patch.items():
            if isinstance(updates, dict):
                if section not in data or not isinstance(data.get(section), dict):
                    data[section] = {}
                # Deep merge: if both the existing value and the incoming value are
                # dicts (e.g. identity.personality_notes), merge recursively so that
                # successive patches accumulate entries rather than overwriting them.
                for key, value in updates.items():
                    if isinstance(value, dict) and not value:
                        continue  # skip empty dicts
                    existing = data[section].get(key)
                    if isinstance(existing, dict) and isinstance(value, dict):
                        existing.update(value)
                    else:
                        data[section][key] = value
            else:
                data[section] = updates
        self.save(data)

        _get_log().info("patch applied — changes: %s", changes)

    def as_yaml_string(self) -> str:
        """Return the full soul file contents as a plain YAML string."""
        return yaml.dump(
            self.load(),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


# ---------------------------------------------------------------------------
# Patch extraction helper
# ---------------------------------------------------------------------------

def _extract_json_patch(text: str) -> Optional[dict]:
    """Parse the first fenced JSON block from text. Returns None if absent or invalid.

    Accepts both ```json ... ``` and plain ``` ... ``` fences so the function is
    robust to models that omit the language tag.
    Falls back to attempting a direct JSON parse of the whole text if no fence
    is found and the text looks like a JSON object.
    """
    # Try ```json ... ``` or ``` ... ``` (language tag optional)
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        # Strip any inner backtick wrappers emitted by some models (```\n`{...}`\n```)
        candidate = match.group(1).strip().strip("`").strip()
    else:
        # No fences — try treating the whole response as raw JSON
        stripped = text.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return None
        candidate = stripped

    try:
        patch = json.loads(candidate)
        return patch if isinstance(patch, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Background patch worker
# ---------------------------------------------------------------------------

def _patch_worker(
    soul: SoulFile,
    conversation_snapshot: list[dict],
    model: str,
    base_url: str,
) -> None:
    """Run in a daemon thread: ask the LLM for a soul patch and apply it if found."""
    _get_log().info("patch_worker start — model=%s, msgs=%d", model, len(conversation_snapshot))
    try:
        # Lazy imports prevent circular dependency issues at module load time
        from src.llm.client import OllamaClient
        from src.llm.prompts import SOUL_PATCH_PROMPT

        # Use the last 12 messages (6 exchanges) so early-conversation facts
        # are still in scope when the patch runs after several turns.
        snippet = conversation_snapshot[-12:]
        conv_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in snippet
        )

        prompt = SOUL_PATCH_PROMPT.replace("{soul_yaml}", soul.as_yaml_string()).replace(
            "{conversation}", conv_text
        )

        # Fresh OllamaClient per thread — requests.Session is not thread-safe to share
        worker_client = OllamaClient(model=model, base_url=base_url)
        raw = worker_client.chat(
            [{"role": "user", "content": prompt}],
            stream=False,
        )

        _get_log().debug("patch_worker raw response (first 500): %s", raw[:500])
        patch = _extract_json_patch(raw)
        if patch:
            # Strip metadata keys added by the updated SOUL_PATCH_PROMPT.
            # Pop before apply_patch so they are never written to the soul file.
            confidence: float = float(patch.pop("_confidence", 1.0))
            explicit: bool = bool(patch.pop("_explicit", True))

            from src.memory.policy import CONFIDENCE_THRESHOLD
            if confidence < CONFIDENCE_THRESHOLD and not explicit:
                _get_log().info(
                    "patch_worker: confidence %.2f below threshold and not explicit "
                    "— skipping patch",
                    confidence,
                )
                return

            _get_log().info(
                "patch_worker applying patch (conf=%.2f, explicit=%s): %s",
                confidence,
                explicit,
                patch,
            )
            soul.apply_patch(patch)
        else:
            _get_log().info("patch_worker: no new facts")

    except Exception as exc:
        _get_log().error("patch_worker error: %s", exc, exc_info=True)


def maybe_update_soul(
    soul: SoulFile,
    conversation: list[dict],
    model: str,
    base_url: str,
) -> None:
    """Spawn a daemon thread to check for soul patches without blocking the main loop."""
    thread = threading.Thread(
        target=_patch_worker,
        args=(soul, list(conversation), model, base_url),
        daemon=True,
    )
    thread.start()


# ---------------------------------------------------------------------------
# Curiosity worker — generates questions Orion wants to ask
# ---------------------------------------------------------------------------

def _extract_question_list(text: str) -> Optional[list[str]]:
    """Parse a {"questions": [...]} block from LLM output. Returns None if absent or invalid."""
    patch = _extract_json_patch(text)
    if not patch:
        return None
    questions = patch.get("questions")
    if not isinstance(questions, list):
        return None
    filtered = [q for q in questions if isinstance(q, str) and q.strip()]
    return filtered if filtered else None


def _curiosity_worker(
    soul: SoulFile,
    conversation_snapshot: list[dict],
    model: str,
    base_url: str,
) -> None:
    """Run in a daemon thread: generate curiosity questions and append to soul."""
    _get_log().info("curiosity_worker start — model=%s, msgs=%d", model, len(conversation_snapshot))
    try:
        from src.llm.client import OllamaClient
        from src.llm.prompts import CURIOSITY_PROMPT

        snippet = conversation_snapshot[-6:]
        conv_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in snippet
        )

        prompt = CURIOSITY_PROMPT.replace("{soul_yaml}", soul.as_yaml_string()).replace(
            "{conversation}", conv_text
        )

        worker_client = OllamaClient(model=model, base_url=base_url)
        raw = worker_client.chat(
            [{"role": "user", "content": prompt}],
            stream=False,
        )

        _get_log().debug("curiosity_worker raw response (first 500): %s", raw[:500])
        new_questions = _extract_question_list(raw)
        if not new_questions:
            _get_log().info("curiosity_worker: no questions found in response")
            return

        _get_log().info("curiosity_worker adding %d question(s): %s", len(new_questions), new_questions)
        # Merge into existing queue, skipping exact duplicates
        data = soul.load()
        identity = data.get("identity") or {}
        existing = identity.get("curiosity_queue") or []
        if not isinstance(existing, list):
            existing = []
        merged = existing + [q for q in new_questions if q not in existing]
        soul.update("identity", "curiosity_queue", merged)

    except Exception as exc:
        _get_log().error("curiosity_worker error: %s", exc, exc_info=True)


def maybe_grow_curiosity(
    soul: SoulFile,
    conversation: list[dict],
    model: str,
    base_url: str,
) -> None:
    """Spawn a daemon thread to generate new curiosity questions without blocking."""
    thread = threading.Thread(
        target=_curiosity_worker,
        args=(soul, list(conversation), model, base_url),
        daemon=True,
    )
    thread.start()


# ---------------------------------------------------------------------------
# Soul → FactsStore migration
# ---------------------------------------------------------------------------

# Keys that STAY in the soul file after migration.
# Everything else in `user` and `partner` is moved to the facts store.
_SOUL_USER_CORE: frozenset[str] = frozenset(
    {"name", "preferred_name", "date_of_birth", "location"}
)
_SOUL_PARTNER_CORE: frozenset[str] = frozenset(
    {"name", "preferred_name", "relationship"}
)

# `identity` and `environment` sections are kept in full — never migrated.


def migrate_soul_to_facts(soul: SoulFile) -> int:
    """Move non-core soul fields into the FactsStore and trim the soul file.

    Only runs when there are fields outside the core sets.  Safe to call
    multiple times — duplicate facts are silently skipped by FactsStore.

    Returns the number of facts added to the store.
    """
    from src.memory.facts_store import FactsStore

    data = soul.load()
    if not data:
        return 0

    facts_store = FactsStore()
    bulk: list[dict] = []

    # --- user section ---
    user = data.get("user")
    if isinstance(user, dict):
        non_core = {k: v for k, v in user.items() if k not in _SOUL_USER_CORE}
        for key, value in non_core.items():
            category = _user_key_to_category(key)
            facts = _flatten_to_facts(key, value, "user", category)
            bulk.extend(facts)
        # Trim user to core keys
        data["user"] = {k: v for k, v in user.items() if k in _SOUL_USER_CORE}

    # --- partner section ---
    partner = data.get("partner")
    if isinstance(partner, dict):
        non_core = {k: v for k, v in partner.items() if k not in _SOUL_PARTNER_CORE}
        for key, value in non_core.items():
            facts = _flatten_to_facts(key, value, "partner", "partner")
            bulk.extend(facts)
        # Trim partner to core keys
        data["partner"] = {k: v for k, v in partner.items() if k in _SOUL_PARTNER_CORE}

    # --- facts section (entire section moves) ---
    facts_section = data.pop("facts", None)
    if isinstance(facts_section, dict):
        for key, value in facts_section.items():
            items = _flatten_to_facts(key, value, "facts", "general")
            bulk.extend(items)

    added = facts_store.add_facts_bulk(bulk)

    # Persist the trimmed soul file
    soul.save(data)
    _get_log().info(
        "migrate_soul_to_facts: migrated %d facts, soul file trimmed", added
    )
    return added


def _user_key_to_category(key: str) -> str:
    """Map a soul user field key to a FactsStore category."""
    _MAP = {
        "profession": "work",
        "education": "education",
        "work_history_summary": "work",
        "personal_projects_highlights": "projects",
        "dungeons_and_dragons": "interests",
        "preferences": "interests",
        "travel": "travel",
        "family": "relationships",
        "email": "general",
        "links": "general",
        "skills": "skills",
        "favorite_band": "interests",
        "favorite_artist": "interests",
        "sports": "interests",
        "music_genres": "interests",
        "pets": "general",
        "teaching": "interests",
    }
    return _MAP.get(key, "general")


def _flatten_to_facts(
    key: str,
    value: object,
    section: str,
    category: str,
) -> list[dict]:
    """Recursively flatten a YAML value into plain-English fact strings."""
    facts: list[dict] = []
    label = key.replace("_", " ")

    if isinstance(value, str):
        facts.append(
            {
                "fact": f"{label}: {value}",
                "category": category,
                "confidence": 1.0,
                "explicit": True,
                "source": f"migration:{section}",
            }
        )

    elif isinstance(value, (int, float, bool)):
        facts.append(
            {
                "fact": f"{label}: {value}",
                "category": category,
                "confidence": 1.0,
                "explicit": True,
                "source": f"migration:{section}",
            }
        )

    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                facts.append(
                    {
                        "fact": f"{label}: {item}",
                        "category": category,
                        "confidence": 1.0,
                        "explicit": True,
                        "source": f"migration:{section}",
                    }
                )
            elif isinstance(item, dict):
                # e.g. education entries, D&D campaign dicts
                summary = "; ".join(
                    f"{k}: {v}" for k, v in item.items() if isinstance(v, str)
                )
                if summary:
                    facts.append(
                        {
                            "fact": f"{label} — {summary}",
                            "category": category,
                            "confidence": 1.0,
                            "explicit": True,
                            "source": f"migration:{section}",
                        }
                    )

    elif isinstance(value, dict):
        for sub_key, sub_val in value.items():
            sub_label = sub_key.replace("_", " ")
            sub_facts = _flatten_to_facts(
                f"{label} — {sub_label}", sub_val, section, category
            )
            facts.extend(sub_facts)

    return facts
