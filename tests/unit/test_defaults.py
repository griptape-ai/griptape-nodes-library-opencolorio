"""Tests for the colour parameter defaults resolver.

These tests run against real OCIO configs loaded with ``CreateFromBuiltinConfig``
rather than the MagicMock convention used elsewhere in this suite. The resolver's
behaviour turns on differences between real config versions, chiefly which File
Rules each one declares, and a mock cannot show that a path matches only the
Default rule on cg-config v1 but a named EXR rule on v4. Loading the four configs
these tests need costs about 20ms in total.

Node and service tests keep using mocks; the distinction is that they exercise
wiring, while these exercise what OCIO itself answers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import PyOpenColorIO as ocio
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from opencolorio.artifacts.ocio_color_params_artifact import OCIOColorParamsArtifact
from opencolorio.defaults import (
    SOURCE_CONFIG_DEFAULT_DISPLAY,
    SOURCE_CONFIG_DEFAULT_RULE,
    SOURCE_CONFIG_DEFAULT_VIEW,
    SOURCE_CONFIG_FILE_RULE,
    SOURCE_CONFIG_ROLE,
    SOURCE_EXPLICIT,
    SOURCE_NO_CONFIG,
    resolve_color_params,
)

CG_V1_URI = "ocio://cg-config-v1.0.0_aces-v1.3_ocio-v2.1"
CG_V2_URI = "ocio://cg-config-v2.2.0_aces-v1.3_ocio-v2.4"
CG_V4_URI = "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5"


@pytest.fixture(scope="module")
def cg_v4_path() -> str:
    return CG_V4_URI


@pytest.fixture(scope="module")
def cg_v1_path() -> str:
    return CG_V1_URI


class TestLayering:
    """Each resolution layer wins over the one below it."""

    def test_explicit_beats_the_config(self, cg_v4_path: str) -> None:
        # Given an explicit display that is not the config's default
        explicit = OCIOColorParamsArtifact(source_colorspace="", display="Rec.1886 Rec.709 - Display", view="")

        # When resolving
        resolved = resolve_color_params(explicit=explicit, config_path=cg_v4_path)

        # Then the explicit value wins and is reported as such
        assert resolved.display == "Rec.1886 Rec.709 - Display"
        assert resolved.sources["display"] == SOURCE_EXPLICIT

    def test_config_answers_when_nothing_explicit(self, cg_v4_path: str) -> None:
        resolved = resolve_color_params(config_path=cg_v4_path)

        config = ocio.Config.CreateFromBuiltinConfig(CG_V4_URI)
        assert resolved.display == config.getDefaultDisplay()
        assert resolved.sources["display"] == SOURCE_CONFIG_DEFAULT_DISPLAY

    def test_working_space_comes_from_the_scene_linear_role(self, cg_v4_path: str) -> None:
        resolved = resolve_color_params(config_path=cg_v4_path)

        assert resolved.working_space == "ACEScg"
        assert resolved.sources["working_space"] == SOURCE_CONFIG_ROLE

    def test_view_comes_from_the_resolved_display(self, cg_v4_path: str) -> None:
        resolved = resolve_color_params(config_path=cg_v4_path)

        config = ocio.Config.CreateFromBuiltinConfig(CG_V4_URI)
        assert resolved.view == config.getDefaultView(config.getDefaultDisplay())
        assert resolved.sources["view"] == SOURCE_CONFIG_DEFAULT_VIEW

    def test_explicit_display_redirects_the_default_view(self, cg_v4_path: str) -> None:
        # Given an explicit display, the view default must follow that display
        explicit = OCIOColorParamsArtifact(source_colorspace="", display="Rec.1886 Rec.709 - Display", view="")

        resolved = resolve_color_params(explicit=explicit, config_path=cg_v4_path)

        config = ocio.Config.CreateFromBuiltinConfig(CG_V4_URI)
        assert resolved.view == config.getDefaultView("Rec.1886 Rec.709 - Display")


class TestNoConfig:
    """$OCIO unset yields the raw config with everything reported as no-config."""

    def test_all_sources_are_no_config(self) -> None:
        resolved = resolve_color_params()

        assert set(resolved.sources.values()) == {SOURCE_NO_CONFIG}

    def test_values_are_empty(self) -> None:
        resolved = resolve_color_params()

        assert resolved.working_space == ""
        assert resolved.source_colorspace == ""
        assert resolved.display == ""
        assert resolved.view == ""

    def test_config_path_is_none(self) -> None:
        assert resolve_color_params().config_path is None

    def test_does_not_raise(self) -> None:
        # The raw config has no scene_linear role; resolution must still complete.
        assert resolve_color_params() is not None

    def test_explicit_values_still_win(self) -> None:
        explicit = OCIOColorParamsArtifact(source_colorspace="ACEScg", display="", view="")

        resolved = resolve_color_params(explicit=explicit)

        assert resolved.source_colorspace == "ACEScg"
        assert resolved.sources["source_colorspace"] == SOURCE_EXPLICIT


class TestConfigAcquisition:
    """A bad config raises; it never degrades into a fallback."""

    def test_missing_file_as_config_path_propagates(self) -> None:
        with pytest.raises(Exception, match="could not read"):
            resolve_color_params(config_path="/nonexistent/nope.ocio")

    def test_missing_file_in_ocio_env_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCIO", "/nonexistent/nope.ocio")

        with pytest.raises(Exception, match="could not read"):
            resolve_color_params()

    def test_ocio_env_is_used_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCIO", CG_V1_URI)

        resolved = resolve_color_params()

        assert resolved.working_space == "ACEScg"
        assert resolved.config_path == CG_V1_URI


class TestFileRules:
    """File Rules drive source_colorspace, and the Default rule is honoured."""

    def test_exr_on_cg_v4_matches_the_named_rule(self, cg_v4_path: str) -> None:
        resolved = resolve_color_params(config_path=cg_v4_path, file_path="/shots/sh010/plate.exr")

        assert resolved.source_colorspace == "ACES2065-1"
        assert resolved.sources["source_colorspace"] == SOURCE_CONFIG_FILE_RULE

    def test_png_on_cg_v4_keeps_the_default_rules_own_value(self, cg_v4_path: str) -> None:
        # The Default rule says sRGB - Display. That is the config author's
        # instruction for unclassified files and must not be substituted.
        resolved = resolve_color_params(config_path=cg_v4_path, file_path="/refs/board.png")

        assert resolved.source_colorspace == "sRGB - Display"
        assert resolved.sources["source_colorspace"] == SOURCE_CONFIG_DEFAULT_RULE

    def test_png_value_is_not_replaced_by_the_working_space(self, cg_v4_path: str) -> None:
        resolved = resolve_color_params(config_path=cg_v4_path, file_path="/refs/board.png")

        assert resolved.source_colorspace != resolved.working_space

    def test_exr_on_cg_v1_resolves_via_its_default_rule(self, cg_v1_path: str) -> None:
        # cg-config v1 declares only a Default rule, so even .exr matches it.
        resolved = resolve_color_params(config_path=cg_v1_path, file_path="/shots/sh010/plate.exr")

        assert resolved.source_colorspace == "ACES2065-1"
        assert resolved.sources["source_colorspace"] == SOURCE_CONFIG_DEFAULT_RULE

    def test_cg_v2_also_has_only_a_default_rule(self) -> None:
        resolved = resolve_color_params(config_path=CG_V2_URI, file_path="/shots/sh010/plate.exr")

        assert resolved.sources["source_colorspace"] == SOURCE_CONFIG_DEFAULT_RULE

    def test_no_file_path_falls_to_the_working_space(self, cg_v4_path: str) -> None:
        resolved = resolve_color_params(config_path=cg_v4_path)

        assert resolved.source_colorspace == resolved.working_space

    def test_no_file_path_is_reported_differently_from_a_default_rule_match(self, cg_v4_path: str) -> None:
        no_path = resolve_color_params(config_path=cg_v4_path)
        default_rule = resolve_color_params(config_path=cg_v4_path, file_path="/refs/board.png")

        assert no_path.sources["source_colorspace"] == SOURCE_CONFIG_ROLE
        assert default_rule.sources["source_colorspace"] == SOURCE_CONFIG_DEFAULT_RULE


class TestRoles:
    """Roles are an open dictionary; no standard role is guaranteed present."""

    def test_config_without_scene_linear_resolves_without_raising(self) -> None:
        # The raw config defines only the "default" role.
        resolved = resolve_color_params()

        assert resolved.sources["working_space"] == SOURCE_NO_CONFIG

    def test_non_standard_role_does_not_break_resolution(self, tmp_path: Path) -> None:
        # Given a config whose only role is one OCIO has never heard of
        config = ocio.Config.CreateRaw()
        config.setRole("studio_bespoke_space", "raw")
        config_file = tmp_path / "bespoke.ocio"
        config_file.write_text(config.serialize())

        # When resolving against it
        resolved = resolve_color_params(config_path=str(config_file))

        # Then it completes, falling through the absent scene_linear role
        assert resolved is not None
        assert resolved.display != "" or resolved.sources["display"] != ""


class TestWarnings:
    """A display-referred source warns; it does not fail."""

    def test_display_referred_source_produces_a_warning(self, cg_v4_path: str) -> None:
        # .png resolves to sRGB - Display, which is display-referred.
        resolved = resolve_color_params(config_path=cg_v4_path, file_path="/refs/board.png")

        assert any("display-referred" in warning for warning in resolved.warnings)

    def test_display_referred_source_still_resolves(self, cg_v4_path: str) -> None:
        resolved = resolve_color_params(config_path=cg_v4_path, file_path="/refs/board.png")

        assert resolved.source_colorspace == "sRGB - Display"

    def test_scene_referred_source_produces_no_warning(self, cg_v4_path: str) -> None:
        resolved = resolve_color_params(config_path=cg_v4_path, file_path="/shots/sh010/plate.exr")

        assert not any("display-referred" in warning for warning in resolved.warnings)


class TestContextVars:
    """Context variables are carried through, not interpreted."""

    def test_context_vars_do_not_change_resolution(self, cg_v4_path: str) -> None:
        without = resolve_color_params(config_path=cg_v4_path)
        with_vars = resolve_color_params(config_path=cg_v4_path, context_vars={"SHOT": "sh010"})

        assert without.display == with_vars.display
        assert without.working_space == with_vars.working_space


class TestStructuralOnly:
    """One test against ocio://default asserting shape, with no names pinned."""

    def test_default_config_fills_every_slot(self) -> None:
        resolved = resolve_color_params(config_path="ocio://default")

        assert resolved.working_space
        assert resolved.source_colorspace
        assert resolved.display
        assert resolved.view

    def test_default_config_reports_a_source_for_every_slot(self) -> None:
        resolved = resolve_color_params(config_path="ocio://default")

        assert set(resolved.sources) == {"working_space", "source_colorspace", "display", "view"}
        assert all(resolved.sources.values())

    def test_result_is_frozen(self) -> None:
        resolved = resolve_color_params(config_path="ocio://default")

        with pytest.raises(Exception, match="cannot assign"):
            resolved.display = "something else"  # type: ignore[misc]
