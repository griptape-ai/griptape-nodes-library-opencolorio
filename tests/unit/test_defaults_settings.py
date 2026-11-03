"""Tests for the library settings layer of the defaults resolver.

Separate from test_defaults.py because every test here patches the settings
reader, and mixing that with the tests that assume no settings invites one
leaking into the other.

The configs are real, for the same reason as in test_defaults.py: a settings
value is only valid or invalid relative to an actual config.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from griptape_nodes.retained_mode.events.config_events import (
    GetConfigValueResultFailure,
    GetConfigValueResultSuccess,
)

from opencolorio.defaults import (
    DEFAULT_SENTINEL,
    SETTING_DISPLAY,
    SETTING_FALLBACK_CONFIG,
    SETTING_VIEW,
    SETTING_WORKING_SPACE,
    SOURCE_CONFIG_DEFAULT_DISPLAY,
    SOURCE_SETTINGS,
    SOURCE_SETTINGS_INVALID,
    resolve_color_params,
)

CG_V4_URI = "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5"


@contextmanager
def settings_patch(**values: str) -> Iterator[None]:
    """Make the named opencolorio settings readable, with every other one empty."""

    def handle_request(request: object) -> object:
        key = str(getattr(request, "category_and_key", "")).removeprefix("opencolorio.")
        if key in values:
            return GetConfigValueResultSuccess(value=values[key], result_details="ok")
        return GetConfigValueResultFailure(result_details="not set")

    mock = MagicMock()
    mock.GriptapeNodes.handle_request.side_effect = handle_request
    with patch("opencolorio.defaults.griptape_nodes", mock):
        yield


class TestSettingsWin:
    """A settings value beats the config's own answer."""

    def test_display_setting_is_used(self) -> None:
        with settings_patch(**{SETTING_DISPLAY: "Rec.1886 Rec.709 - Display"}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.display == "Rec.1886 Rec.709 - Display"
        assert resolved.sources["display"] == SOURCE_SETTINGS

    def test_working_space_setting_is_used(self) -> None:
        with settings_patch(**{SETTING_WORKING_SPACE: "ACES2065-1"}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.working_space == "ACES2065-1"
        assert resolved.sources["working_space"] == SOURCE_SETTINGS

    def test_a_role_name_is_a_valid_working_space_setting(self) -> None:
        with settings_patch(**{SETTING_WORKING_SPACE: "compositing_log"}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.working_space == "compositing_log"
        assert resolved.sources["working_space"] == SOURCE_SETTINGS

    def test_view_setting_is_validated_against_the_resolved_display(self) -> None:
        with settings_patch(**{SETTING_DISPLAY: "Rec.1886 Rec.709 - Display", SETTING_VIEW: "Raw"}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.view == "Raw"
        assert resolved.sources["view"] == SOURCE_SETTINGS

    def test_explicit_still_beats_a_setting(self) -> None:
        from opencolorio.artifacts.ocio_color_params_artifact import OCIOColorParamsArtifact

        explicit = OCIOColorParamsArtifact(source_colorspace="", display="sRGB - Display", view="")
        with settings_patch(**{SETTING_DISPLAY: "Rec.1886 Rec.709 - Display"}):
            resolved = resolve_color_params(explicit=explicit, config_path=CG_V4_URI)

        assert resolved.display == "sRGB - Display"


class TestInvalidSettings:
    """A typo falls through to the config rather than raising at render time."""

    def test_invalid_display_falls_through_to_the_config(self) -> None:
        with settings_patch(**{SETTING_DISPLAY: "No Such Display"}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.display == "sRGB - Display"

    def test_invalid_display_is_reported_as_settings_invalid(self) -> None:
        with settings_patch(**{SETTING_DISPLAY: "No Such Display"}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.sources["display"] == SOURCE_SETTINGS_INVALID

    def test_invalid_display_warns_naming_the_value(self) -> None:
        with settings_patch(**{SETTING_DISPLAY: "No Such Display"}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert any("No Such Display" in warning for warning in resolved.warnings)

    def test_invalid_working_space_falls_through(self) -> None:
        with settings_patch(**{SETTING_WORKING_SPACE: "Nonsense Space"}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.working_space == "ACEScg"
        assert resolved.sources["working_space"] == SOURCE_SETTINGS_INVALID

    def test_invalid_view_falls_through(self) -> None:
        with settings_patch(**{SETTING_VIEW: "No Such View"}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.sources["view"] == SOURCE_SETTINGS_INVALID
        assert resolved.view != ""

    def test_an_invalid_setting_does_not_raise(self) -> None:
        with settings_patch(**{SETTING_DISPLAY: "No Such Display", SETTING_WORKING_SPACE: "Nope"}):
            assert resolve_color_params(config_path=CG_V4_URI) is not None


class TestFallbackConfig:
    """fallback_config fires only when $OCIO is unset, and takes ocio:// only."""

    def test_builtin_uri_is_used_when_ocio_is_unset(self) -> None:
        with settings_patch(**{SETTING_FALLBACK_CONFIG: CG_V4_URI}):
            resolved = resolve_color_params()

        assert resolved.config_path == CG_V4_URI
        assert resolved.working_space == "ACEScg"

    def test_builtin_uri_means_slots_are_not_no_config(self) -> None:
        with settings_patch(**{SETTING_FALLBACK_CONFIG: CG_V4_URI}):
            resolved = resolve_color_params()

        assert resolved.sources["display"] == SOURCE_CONFIG_DEFAULT_DISPLAY

    def test_a_filesystem_path_is_rejected_with_a_warning(self) -> None:
        with settings_patch(**{SETTING_FALLBACK_CONFIG: "/studio/configs/show.ocio"}):
            resolved = resolve_color_params()

        assert any("/studio/configs/show.ocio" in warning for warning in resolved.warnings)

    def test_a_filesystem_path_degrades_to_the_raw_config(self) -> None:
        with settings_patch(**{SETTING_FALLBACK_CONFIG: "/studio/configs/show.ocio"}):
            resolved = resolve_color_params()

        assert resolved.config_path is None
        assert resolved.display == ""

    def test_an_unknown_builtin_uri_warns_and_degrades(self) -> None:
        with settings_patch(**{SETTING_FALLBACK_CONFIG: "ocio://no-such-config"}):
            resolved = resolve_color_params()

        assert any("no-such-config" in warning for warning in resolved.warnings)
        assert resolved.config_path is None

    def test_a_bad_value_does_not_raise(self) -> None:
        with settings_patch(**{SETTING_FALLBACK_CONFIG: "ocio://no-such-config"}):
            assert resolve_color_params() is not None

    def test_it_is_ignored_when_ocio_is_set_but_broken(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A config the user pointed at and that will not load stays an error.
        monkeypatch.setenv("OCIO", "/nonexistent/nope.ocio")

        with settings_patch(**{SETTING_FALLBACK_CONFIG: CG_V4_URI}), pytest.raises(Exception, match="could not read"):
            resolve_color_params()

    def test_it_is_ignored_when_an_explicit_config_path_is_given(self) -> None:
        with settings_patch(**{SETTING_FALLBACK_CONFIG: CG_V4_URI}):
            resolved = resolve_color_params(config_path="ocio://cg-config-v1.0.0_aces-v1.3_ocio-v2.1")

        assert resolved.config_path == "ocio://cg-config-v1.0.0_aces-v1.3_ocio-v2.1"


class TestSettingsReader:
    """Reading settings goes through the engine config request."""

    def test_a_missing_key_reads_as_empty(self) -> None:
        with settings_patch():
            resolved = resolve_color_params(config_path=CG_V4_URI)

        # Falls to the config's own answer, not to a settings value.
        assert resolved.sources["display"] == SOURCE_CONFIG_DEFAULT_DISPLAY

    def test_values_are_stripped(self) -> None:
        with settings_patch(**{SETTING_DISPLAY: "  sRGB - Display  "}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.display == "sRGB - Display"
        assert resolved.sources["display"] == SOURCE_SETTINGS

    def test_the_request_names_the_opencolorio_category(self) -> None:
        seen: list[str] = []

        def handle_request(request: object) -> object:
            seen.append(str(getattr(request, "category_and_key", "")))
            return GetConfigValueResultFailure(result_details="not set")

        mock = MagicMock()
        mock.GriptapeNodes.handle_request.side_effect = handle_request
        with patch("opencolorio.defaults.griptape_nodes", mock):
            resolve_color_params(config_path=CG_V4_URI)

        assert "opencolorio.default_display" in seen
        assert "opencolorio.fallback_config" in seen

    def test_no_engine_running_reads_as_empty(self) -> None:
        mock = MagicMock()
        mock.GriptapeNodes.handle_request.side_effect = RuntimeError("no engine")
        with patch("opencolorio.defaults.griptape_nodes", mock):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.sources["display"] == SOURCE_CONFIG_DEFAULT_DISPLAY


class TestSentinelSettingValue:
    """The shipped value is `<default>`, and it means the same as unset.

    The engine drops blank strings as each config layer loads, on the basis that
    a layer holding "" states nothing about that setting. Shipping all four as ""
    therefore left the category empty, and the editor, which iterates the config
    contents rather than the schema, had no rows to draw and hid the section.
    `<default>` survives the loader while still meaning "not chosen".
    """

    def test_the_sentinel_is_treated_as_unset(self) -> None:
        with settings_patch(**{SETTING_DISPLAY: DEFAULT_SENTINEL}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.display == "sRGB - Display"
        assert resolved.sources["display"] == SOURCE_CONFIG_DEFAULT_DISPLAY

    def test_the_sentinel_does_not_warn_as_invalid(self) -> None:
        with settings_patch(**{SETTING_WORKING_SPACE: DEFAULT_SENTINEL}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.sources["working_space"] != SOURCE_SETTINGS_INVALID
        assert not any(DEFAULT_SENTINEL in warning for warning in resolved.warnings)

    def test_a_sentinel_fallback_config_is_ignored(self) -> None:
        # It must not be mistaken for a bad ocio:// URI and warned about.
        with settings_patch(**{SETTING_FALLBACK_CONFIG: DEFAULT_SENTINEL}):
            resolved = resolve_color_params()

        assert resolved.config_path is None
        assert not any(DEFAULT_SENTINEL in warning for warning in resolved.warnings)

    def test_a_real_value_still_wins(self) -> None:
        with settings_patch(**{SETTING_DISPLAY: "Rec.1886 Rec.709 - Display"}):
            resolved = resolve_color_params(config_path=CG_V4_URI)

        assert resolved.sources["display"] == SOURCE_SETTINGS
