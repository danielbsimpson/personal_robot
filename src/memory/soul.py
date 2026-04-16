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
# Worker logging — writes to data/soul_worker.log
# ---------------------------------------------------------------------------

SOUL_LOG_PATH = _PROJECT_ROOT / "data" / "soul_worker.log"

_log = logging.getLogger("soul_worker")
if not _log.handlers:
    _log.setLevel(logging.DEBUG)
    try:
        SOUL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(SOUL_LOG_PATH, encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
        _log.addHandler(_fh)
    except OSError:
        pass


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

    def to_prompt_section(self) -> str:
        """Return a formatted ## About Me YAML block for system prompt injection."""
        data = self.load()
        if not data:
            return ""
        soul_yaml = yaml.dump(
            data, default_flow_style=False, allow_unicode=True, sort_keys=False
        )
        return f"## About Me\n\n```yaml\n{soul_yaml}```"

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
        for section, updates in patch.items():
            if isinstance(updates, dict):
                if section not in data or not isinstance(data.get(section), dict):
                    data[section] = {}
                data[section].update(updates)
            else:
                data[section] = updates
        self.save(data)

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
    """Parse the first ```json ... ``` block in text. Returns None if absent or invalid."""
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        return None
    try:
        patch = json.loads(match.group(1))
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
    _log.info("patch_worker start — model=%s, msgs=%d", model, len(conversation_snapshot))
    try:
        # Lazy imports prevent circular dependency issues at module load time
        from src.llm.client import OllamaClient
        from src.llm.prompts import SOUL_PATCH_PROMPT

        # Use only the last 6 messages (3 exchanges) to keep the prompt compact
        snippet = conversation_snapshot[-6:]
        conv_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in snippet
        )

        prompt = SOUL_PATCH_PROMPT.format(
            soul_yaml=soul.as_yaml_string(),
            conversation=conv_text,
        )

        # Fresh OllamaClient per thread — requests.Session is not thread-safe to share
        worker_client = OllamaClient(model=model, base_url=base_url)
        raw = worker_client.chat(
            [{"role": "user", "content": prompt}],
            stream=False,
        )

        _log.debug("patch_worker raw response (first 500): %s", raw[:500])
        patch = _extract_json_patch(raw)
        if patch:
            _log.info("patch_worker applying patch: %s", patch)
            soul.apply_patch(patch)
        else:
            _log.info("patch_worker: no patch — LLM found nothing new to record")

    except Exception as exc:
        _log.error("patch_worker error: %s", exc, exc_info=True)


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
    _log.info("curiosity_worker start — model=%s, msgs=%d", model, len(conversation_snapshot))
    try:
        from src.llm.client import OllamaClient
        from src.llm.prompts import CURIOSITY_PROMPT

        snippet = conversation_snapshot[-6:]
        conv_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in snippet
        )

        prompt = CURIOSITY_PROMPT.format(
            soul_yaml=soul.as_yaml_string(),
            conversation=conv_text,
        )

        worker_client = OllamaClient(model=model, base_url=base_url)
        raw = worker_client.chat(
            [{"role": "user", "content": prompt}],
            stream=False,
        )

        _log.debug("curiosity_worker raw response (first 500): %s", raw[:500])
        new_questions = _extract_question_list(raw)
        if not new_questions:
            _log.info("curiosity_worker: no questions found in response")
            return

        _log.info("curiosity_worker adding %d question(s): %s", len(new_questions), new_questions)
        # Merge into existing queue, skipping exact duplicates
        data = soul.load()
        identity = data.get("identity") or {}
        existing = identity.get("curiosity_queue") or []
        if not isinstance(existing, list):
            existing = []
        merged = existing + [q for q in new_questions if q not in existing]
        soul.update("identity", "curiosity_queue", merged)

    except Exception as exc:
        _log.error("curiosity_worker error: %s", exc, exc_info=True)


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
