"""Tests for the shape of griptape-nodes-library.json.

The editor's Library settings page renders top-level categories with scalar
string values and skips nested dicts, so the shape here is what decides whether
a user ever sees these settings. It is also worth pinning because library
settings are merged into every user's global config at registration and are
never pruned, so a key shipped and later renamed is left behind for good.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "griptape-nodes-library.json"

EXPECTED_SETTING_KEYS = {
    "default_working_space",
    "default_display",
    "default_view",
    "fallback_config",
}


def _manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def _opencolorio_setting() -> dict[str, Any]:
    for setting in _manifest()["settings"]:
        if setting["category"] == "opencolorio":
            return setting
    msg = "No settings entry with category 'opencolorio'"
    raise AssertionError(msg)


class TestOpenColorIOSettings:
    def test_the_category_exists(self) -> None:
        assert _opencolorio_setting()["category"] == "opencolorio"

    def test_it_declares_exactly_the_expected_keys(self) -> None:
        assert set(_opencolorio_setting()["contents"]) == EXPECTED_SETTING_KEYS

    def test_every_value_ships_empty(self) -> None:
        # Installing the library must change nothing for an existing workflow.
        assert all(value == "" for value in _opencolorio_setting()["contents"].values())

    def test_every_value_is_a_scalar_string(self) -> None:
        # The editor skips nested dicts, so a nested value would never render.
        assert all(isinstance(value, str) for value in _opencolorio_setting()["contents"].values())

    def test_every_key_has_a_schema_entry(self) -> None:
        schema = _opencolorio_setting()["schema"]
        assert set(schema) == EXPECTED_SETTING_KEYS

    def test_every_schema_entry_has_a_title_and_description(self) -> None:
        for key, entry in _opencolorio_setting()["schema"].items():
            assert entry.get("type") == "string", key
            assert entry.get("title"), key
            assert entry.get("description"), key

    def test_the_description_says_what_the_category_is_for(self) -> None:
        assert _opencolorio_setting()["description"]


class TestSecretsSettingIsUntouched:
    def test_the_ocio_secret_is_still_registered(self) -> None:
        categories = [setting["category"] for setting in _manifest()["settings"]]
        assert "app_events.on_app_initialization_complete" in categories

    def test_the_ocio_secret_still_ships_empty(self) -> None:
        for setting in _manifest()["settings"]:
            if setting["category"] == "app_events.on_app_initialization_complete":
                assert setting["contents"]["secrets_to_register"] == {"OCIO": ""}
                return
        msg = "The OCIO secret registration is gone"
        raise AssertionError(msg)
