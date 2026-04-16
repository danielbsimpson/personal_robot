"""
Pytest configuration — adds the project root to sys.path so that
`src.*` imports work without needing `pip install -e .`.
"""
import logging
import sys
import os
from pathlib import Path

import pytest

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def log_dir(tmp_path: Path, monkeypatch) -> Path:
    """Redirect all data/logs/ output to a per-test temp directory.

    Patches _DEFAULT_LOGS_DIR in src.utils.log, then clears any already-
    initialised handlers for the loggers that Phase 1.8 uses so they will be
    recreated pointing at the temp directory on next use.
    """
    logs_path = tmp_path / "logs"

    import src.utils.log as log_module
    monkeypatch.setattr(log_module, "_DEFAULT_LOGS_DIR", logs_path)

    # Clear cached handlers so the loggers are recreated with the new path
    for logger_name in (
        "personal_robot.soul_changes",
        "personal_robot.context_trim",
        "personal_robot.memory",
    ):
        cached = logging.getLogger(logger_name)
        cached.handlers.clear()

    # Reset lazy logger references so modules call get_logger() again
    import src.memory.soul as soul_module
    import src.llm.client as client_module
    monkeypatch.setattr(soul_module, "_log", None)
    monkeypatch.setattr(client_module, "_trim_log", None)

    return logs_path
