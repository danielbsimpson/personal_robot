"""
Central logging utilities for Personal Robot.

get_logger(name)      — Returns a RotatingFileHandler logger writing to
                        data/logs/<name>.log (5 MB × 3 backups). WARNING+
                        messages also go to stderr. Safe to call multiple
                        times with the same name — handlers are only added once.

ConversationLogger    — Appends user/assistant turns as JSONL to
                        data/logs/conversations/<YYYY-MM-DD_HH-MM-SS>.jsonl.
                        A new file is created per instance (one per session).
"""

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Exposed at module level so tests can monkeypatch it before loggers are created.
_DEFAULT_LOGS_DIR: Path = _PROJECT_ROOT / "data" / "logs"


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

def get_logger(name: str, logs_dir: Path | None = None) -> logging.Logger:
    """Return a named logger writing to logs_dir/<name>.log.

    Uses a RotatingFileHandler (5 MB × 3 backups). WARNING+ messages also
    go to stderr. Safe to call multiple times — handlers are only added once.

    Args:
        name:     Logger name; also used as the log file stem.
        logs_dir: Directory for log files. Defaults to data/logs/.

    Returns:
        Configured logging.Logger instance.
    """
    if logs_dir is None:
        logs_dir = _DEFAULT_LOGS_DIR

    logger = logging.getLogger(f"personal_robot.{name}")
    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            logs_dir / f"{name}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
        )
        logger.addHandler(fh)
    except OSError:
        pass  # silently fall back to console-only if the log dir is inaccessible

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(
        logging.Formatter("%(levelname)s  %(name)s — %(message)s")
    )
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# ConversationLogger
# ---------------------------------------------------------------------------

class ConversationLogger:
    """Appends conversation turns as JSONL to one file per session.

    File path: data/logs/conversations/<YYYY-MM-DD_HH-MM-SS>.jsonl

    A new file is created each time this class is instantiated, so each
    CLI run or Streamlit session gets its own log file.
    """

    def __init__(self, model: str, logs_dir: Path | None = None) -> None:
        if logs_dir is None:
            logs_dir = _DEFAULT_LOGS_DIR
        self.model = model
        session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        conv_dir = logs_dir / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        self._path = conv_dir / f"{session_ts}.jsonl"

    @property
    def path(self) -> Path:
        """Path of the JSONL file for this session."""
        return self._path

    def log_turn(self, role: str, content: str, model: str | None = None) -> None:
        """Append one message as a JSON line.

        Args:
            role:    "user" or "assistant".
            content: The full message text.
            model:   Optional model name override; falls back to the instance model.
        """
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "content": content,
            "model": model or self.model,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
