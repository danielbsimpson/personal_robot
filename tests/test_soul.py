"""
Tests for src/memory/soul.py

Uses pytest's tmp_path fixture so tests never touch the real data/soul.yaml.
"""

import json
from pathlib import Path

import pytest
import yaml

from src.memory.soul import SoulFile, _extract_json_patch, _extract_question_list, maybe_grow_curiosity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def soul(tmp_path: Path) -> SoulFile:
    """Return a SoulFile pointing to a fresh temp directory."""
    return SoulFile(path=tmp_path / "soul.yaml")


@pytest.fixture
def seeded_soul(tmp_path: Path) -> SoulFile:
    """Return a SoulFile pre-populated with known data."""
    data = {
        "identity": {"name": "Orion"},
        "user": {"name": "Daniel"},
        "facts": {},
    }
    soul_path = tmp_path / "soul.yaml"
    with soul_path.open("w") as f:
        yaml.dump(data, f)
    return SoulFile(path=soul_path)


# ---------------------------------------------------------------------------
# SoulFile.load
# ---------------------------------------------------------------------------

def test_load_returns_empty_dict_when_file_missing(soul: SoulFile) -> None:
    assert soul.load() == {}


def test_load_returns_parsed_yaml(seeded_soul: SoulFile) -> None:
    data = seeded_soul.load()
    assert data["identity"]["name"] == "Orion"
    assert data["user"]["name"] == "Daniel"


# ---------------------------------------------------------------------------
# SoulFile.save / SoulFile.load round-trip
# ---------------------------------------------------------------------------

def test_save_creates_file_and_reloads(soul: SoulFile) -> None:
    soul.save({"identity": {"name": "Orion"}})
    assert soul.path.exists()
    assert soul.load()["identity"]["name"] == "Orion"


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    nested = SoulFile(path=tmp_path / "deep" / "nested" / "soul.yaml")
    nested.save({"facts": {"x": "y"}})
    assert nested.path.exists()


# ---------------------------------------------------------------------------
# SoulFile.update
# ---------------------------------------------------------------------------

def test_update_sets_key_in_existing_section(seeded_soul: SoulFile) -> None:
    seeded_soul.update("user", "job", "engineer")
    assert seeded_soul.load()["user"]["job"] == "engineer"


def test_update_creates_new_section(soul: SoulFile) -> None:
    soul.save({})
    soul.update("environment", "location", "home office")
    assert soul.load()["environment"]["location"] == "home office"


def test_update_preserves_existing_keys(seeded_soul: SoulFile) -> None:
    seeded_soul.update("user", "job", "engineer")
    data = seeded_soul.load()
    # Original name should still be there
    assert data["user"]["name"] == "Daniel"
    assert data["user"]["job"] == "engineer"


# ---------------------------------------------------------------------------
# SoulFile.apply_patch
# ---------------------------------------------------------------------------

def test_apply_patch_merges_into_existing_section(seeded_soul: SoulFile) -> None:
    seeded_soul.apply_patch({"user": {"hobby": "robotics"}})
    data = seeded_soul.load()
    assert data["user"]["hobby"] == "robotics"
    assert data["user"]["name"] == "Daniel"  # existing key preserved


def test_apply_patch_creates_new_section(soul: SoulFile) -> None:
    soul.save({})
    soul.apply_patch({"facts": {"favourite_food": "pizza"}})
    assert soul.load()["facts"]["favourite_food"] == "pizza"


def test_apply_patch_with_multiple_sections(seeded_soul: SoulFile) -> None:
    seeded_soul.apply_patch({
        "user": {"job": "developer"},
        "facts": {"city": "London"},
    })
    data = seeded_soul.load()
    assert data["user"]["job"] == "developer"
    assert data["facts"]["city"] == "London"


def test_apply_patch_empty_patch_is_noop(seeded_soul: SoulFile) -> None:
    before = seeded_soul.load()
    seeded_soul.apply_patch({})
    assert seeded_soul.load() == before


# ---------------------------------------------------------------------------
# SoulFile.to_prompt_section
# ---------------------------------------------------------------------------

def test_to_prompt_section_contains_header(seeded_soul: SoulFile) -> None:
    section = seeded_soul.to_prompt_section()
    assert "## About Me" in section


def test_to_prompt_section_contains_identity_name(seeded_soul: SoulFile) -> None:
    section = seeded_soul.to_prompt_section()
    assert "Orion" in section


def test_to_prompt_section_returns_empty_string_when_no_file(soul: SoulFile) -> None:
    assert soul.to_prompt_section() == ""


# ---------------------------------------------------------------------------
# SoulFile.as_yaml_string
# ---------------------------------------------------------------------------

def test_as_yaml_string_is_parseable(seeded_soul: SoulFile) -> None:
    raw = seeded_soul.as_yaml_string()
    parsed = yaml.safe_load(raw)
    assert parsed["identity"]["name"] == "Orion"


# ---------------------------------------------------------------------------
# _extract_json_patch
# ---------------------------------------------------------------------------

def test_extract_json_patch_valid_block() -> None:
    text = '```json\n{"user": {"job": "engineer"}}\n```'
    patch = _extract_json_patch(text)
    assert patch == {"user": {"job": "engineer"}}


def test_extract_json_patch_multiline_json() -> None:
    text = '```json\n{\n  "user": {\n    "job": "engineer"\n  }\n}\n```'
    patch = _extract_json_patch(text)
    assert patch is not None
    assert patch["user"]["job"] == "engineer"


def test_extract_json_patch_no_block_returns_none() -> None:
    assert _extract_json_patch("Nothing useful here.") is None


def test_extract_json_patch_invalid_json_returns_none() -> None:
    assert _extract_json_patch("```json\n{invalid}\n```") is None


def test_extract_json_patch_non_dict_returns_none() -> None:
    assert _extract_json_patch('```json\n["list", "not", "dict"]\n```') is None


def test_extract_json_patch_surrounded_by_text() -> None:
    text = "Sure, I'll remember that!\n```json\n{\"facts\": {\"pet\": \"dog\"}}\n```\nDone."
    patch = _extract_json_patch(text)
    assert patch == {"facts": {"pet": "dog"}}


# ---------------------------------------------------------------------------
# _extract_question_list
# ---------------------------------------------------------------------------

def test_extract_question_list_valid_block() -> None:
    text = '```json\n{"questions": ["What music does Danielle like?", "How long have you had Orion?"]}\n```'
    result = _extract_question_list(text)
    assert result == ["What music does Danielle like?", "How long have you had Orion?"]


def test_extract_question_list_no_block_returns_none() -> None:
    assert _extract_question_list("Nothing useful here.") is None


def test_extract_question_list_missing_key_returns_none() -> None:
    text = '```json\n{"facts": {"x": "y"}}\n```'
    assert _extract_question_list(text) is None


def test_extract_question_list_filters_non_strings() -> None:
    text = '```json\n{"questions": ["valid question", 42, null, "another valid"]}\n```'
    result = _extract_question_list(text)
    assert result == ["valid question", "another valid"]


def test_extract_question_list_all_invalid_returns_none() -> None:
    text = '```json\n{"questions": [42, null, false]}\n```'
    assert _extract_question_list(text) is None


# ---------------------------------------------------------------------------
# maybe_grow_curiosity
# ---------------------------------------------------------------------------

def test_maybe_grow_curiosity_spawns_thread(tmp_path: Path, monkeypatch) -> None:
    """maybe_grow_curiosity should return immediately (daemon thread, non-blocking)."""
    import threading

    spawned = []
    original_start = threading.Thread.start

    def mock_start(self):
        spawned.append(self)
        # Don't actually start the thread (avoids needing Ollama in tests)

    monkeypatch.setattr(threading.Thread, "start", mock_start)

    soul = SoulFile(path=tmp_path / "soul.yaml")
    soul.save({"identity": {"name": "Orion"}, "facts": {}})

    maybe_grow_curiosity(soul, [{"role": "user", "content": "hello"}], "phi4-mini", "http://localhost:11434")

    assert len(spawned) == 1
    assert spawned[0].daemon is True
