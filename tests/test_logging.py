"""
Tests for src/utils/log.py (Phase 1.8.7)

Covers:
  - get_logger: file creation, singleton behaviour, message content
  - ConversationLogger: JSONL creation, valid JSON per line, correct fields
  - Soul audit log: apply_patch() emits an audit entry
  - Context trim log: trim_history() logs dropped messages

All tests use tmp_path fixtures and never touch production log files.
"""

import json
import logging
from pathlib import Path

import pytest
import yaml

from src.utils.log import ConversationLogger, get_logger


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_creates_log_file(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logger = get_logger("test_create", logs_dir=logs)
        logger.info("hello")
        for h in logger.handlers:
            h.flush()
        assert (logs / "test_create.log").exists()

    def test_message_written_to_file(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logger = get_logger("test_write", logs_dir=logs)
        logger.info("unique_marker_xyz")
        for h in logger.handlers:
            h.flush()
        content = (logs / "test_write.log").read_text(encoding="utf-8")
        assert "unique_marker_xyz" in content

    def test_returns_same_instance_on_repeat_call(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        a = get_logger("test_singleton", logs_dir=logs)
        b = get_logger("test_singleton", logs_dir=logs)
        assert a is b

    def test_does_not_add_duplicate_handlers(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logger = get_logger("test_no_dup", logs_dir=logs)
        initial_count = len(logger.handlers)
        # Calling again should be a no-op because handlers already exist
        get_logger("test_no_dup", logs_dir=logs)
        assert len(logger.handlers) == initial_count


# ---------------------------------------------------------------------------
# ConversationLogger
# ---------------------------------------------------------------------------

class TestConversationLogger:
    def test_creates_jsonl_file(self, tmp_path: Path) -> None:
        cl = ConversationLogger(model="phi4-mini", logs_dir=tmp_path)
        cl.log_turn("user", "hello")
        assert cl.path.exists()

    def test_file_in_conversations_subdir(self, tmp_path: Path) -> None:
        cl = ConversationLogger(model="phi4-mini", logs_dir=tmp_path)
        cl.log_turn("user", "hello")
        assert cl.path.parent.name == "conversations"

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        cl = ConversationLogger(model="phi4-mini", logs_dir=tmp_path)
        cl.log_turn("user", "ping")
        cl.log_turn("assistant", "pong")
        lines = cl.path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)  # raises if invalid
            assert isinstance(record, dict)

    def test_required_fields_present(self, tmp_path: Path) -> None:
        cl = ConversationLogger(model="test-model", logs_dir=tmp_path)
        cl.log_turn("user", "content here")
        record = json.loads(cl.path.read_text(encoding="utf-8").strip())
        assert record["role"] == "user"
        assert record["content"] == "content here"
        assert record["model"] == "test-model"
        assert "ts" in record

    def test_model_override_per_turn(self, tmp_path: Path) -> None:
        cl = ConversationLogger(model="default-model", logs_dir=tmp_path)
        cl.log_turn("user", "hi", model="override-model")
        record = json.loads(cl.path.read_text(encoding="utf-8").strip())
        assert record["model"] == "override-model"

    def test_fallback_to_instance_model(self, tmp_path: Path) -> None:
        cl = ConversationLogger(model="instance-model", logs_dir=tmp_path)
        cl.log_turn("assistant", "response")
        record = json.loads(cl.path.read_text(encoding="utf-8").strip())
        assert record["model"] == "instance-model"

    def test_multiple_turns_accumulate(self, tmp_path: Path) -> None:
        cl = ConversationLogger(model="phi4-mini", logs_dir=tmp_path)
        for i in range(5):
            cl.log_turn("user" if i % 2 == 0 else "assistant", f"message {i}")
        lines = cl.path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# Soul audit log via apply_patch()
# ---------------------------------------------------------------------------

class TestSoulAuditLog:
    """Verify that apply_patch() emits an audit entry to soul_changes.log."""

    def test_apply_patch_writes_audit_entry(
        self, tmp_path: Path, log_dir: Path
    ) -> None:
        from src.memory.soul import SoulFile

        soul_path = tmp_path / "soul.yaml"
        with soul_path.open("w") as f:
            yaml.dump({"identity": {"name": "Orion"}}, f)

        soul = SoulFile(path=soul_path)
        soul.apply_patch({"identity": {"name": "Orion v2"}})

        log_file = log_dir / "soul_changes.log"
        assert log_file.exists(), "soul_changes.log was not created"
        content = log_file.read_text(encoding="utf-8")
        assert "patch applied" in content

    def test_audit_entry_contains_changed_key(
        self, tmp_path: Path, log_dir: Path
    ) -> None:
        from src.memory.soul import SoulFile

        soul_path = tmp_path / "soul.yaml"
        with soul_path.open("w") as f:
            yaml.dump({"user": {"name": "Daniel"}}, f)

        soul = SoulFile(path=soul_path)
        soul.apply_patch({"user": {"hobby": "cycling"}})

        log_file = log_dir / "soul_changes.log"
        content = log_file.read_text(encoding="utf-8")
        assert "user.hobby" in content

    def test_new_key_recorded_as_new(
        self, tmp_path: Path, log_dir: Path
    ) -> None:
        from src.memory.soul import SoulFile

        soul_path = tmp_path / "soul.yaml"
        with soul_path.open("w") as f:
            yaml.dump({}, f)

        soul = SoulFile(path=soul_path)
        soul.apply_patch({"facts": {"coffee": "prefers oat milk"}})

        log_file = log_dir / "soul_changes.log"
        content = log_file.read_text(encoding="utf-8")
        assert "<new>" in content


# ---------------------------------------------------------------------------
# Context trim log via trim_history()
# ---------------------------------------------------------------------------

class TestContextTrimLog:
    """Verify that trim_history() logs dropped messages to context_trim.log."""

    def _make_messages(self, n_pairs: int, chars_each: int = 500) -> list[dict]:
        messages = []
        for i in range(n_pairs):
            messages.append({"role": "user", "content": "u" * chars_each})
            messages.append({"role": "assistant", "content": "a" * chars_each})
        return messages

    def test_trim_creates_log_when_messages_dropped(
        self, log_dir: Path
    ) -> None:
        from src.llm.client import trim_history

        msgs = self._make_messages(4, chars_each=500)  # 4000 chars total
        trim_history(msgs, limit_chars=1000)

        log_file = log_dir / "context_trim.log"
        assert log_file.exists(), "context_trim.log was not created"
        content = log_file.read_text(encoding="utf-8")
        assert "trimmed" in content

    def test_trim_no_log_when_nothing_dropped(
        self, log_dir: Path
    ) -> None:
        from src.llm.client import trim_history

        msgs = self._make_messages(1, chars_each=10)  # 20 chars — well within budget
        trim_history(msgs, limit_chars=1000)

        log_file = log_dir / "context_trim.log"
        # File may not exist at all, or exist but empty — neither is okay only if it has content
        if log_file.exists():
            assert log_file.read_text(encoding="utf-8") == ""

    def test_trim_records_drop_counts(
        self, log_dir: Path
    ) -> None:
        from src.llm.client import trim_history

        msgs = self._make_messages(3, chars_each=300)  # 1800 chars
        trim_history(msgs, limit_chars=700)

        content = (log_dir / "context_trim.log").read_text(encoding="utf-8")
        # Should record before→after counts
        assert "→" in content

    def test_trim_always_keeps_most_recent(
        self, log_dir: Path
    ) -> None:
        from src.llm.client import trim_history

        last_content = "keep_this_message"
        msgs = self._make_messages(3, chars_each=300)
        msgs[-1]["content"] = last_content  # replace last message content
        result = trim_history(msgs, limit_chars=200)

        assert any(last_content in m["content"] for m in result)
