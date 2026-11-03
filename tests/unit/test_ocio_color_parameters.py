from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from griptape_nodes.traits.options import Options

from opencolorio.artifacts.ocio_color_params_artifact import OCIOColorParamsArtifact
from opencolorio.artifacts.ocio_config_artifact import OCIOConfigArtifact
from opencolorio.defaults import DEFAULT_SENTINEL, ResolvedColorParams
from opencolorio.nodes.config.ocio_color_parameters import OCIOColorParameters


def _make_node() -> OCIOColorParameters:
    return OCIOColorParameters("test_node")


def _make_config_artifact(file_path: str | None = None) -> OCIOConfigArtifact:
    return OCIOConfigArtifact(file_path=file_path)


def _mock_ocio_config(
    colorspaces: list[str] | None = None,
    displays: list[str] | None = None,
    views_by_display: dict[str, list[str]] | None = None,
    roles: list[tuple[str, str]] | None = None,
) -> MagicMock:
    """Build a mock ocio.Config with controllable colorspace/display/view/role lists."""
    colorspaces = colorspaces or ["ACEScg", "scene_linear", "sRGB"]
    displays = displays or ["ACES", "sRGB"]
    views_by_display = views_by_display or {"ACES": ["sRGB", "Log"], "sRGB": ["Raw", "Film"]}
    roles = roles or [("scene_linear", "ACEScg"), ("compositing_log", "ACEScc")]

    mock_config = MagicMock()
    mock_config.getColorSpaceNames.return_value = colorspaces
    mock_config.getDisplays.return_value = displays
    mock_config.getViews.side_effect = lambda display: views_by_display.get(display, [])
    mock_config.getRoles.return_value = roles
    return mock_config


class TestNodeInstantiation:
    def test_node_can_be_created_without_config(self) -> None:
        node = _make_node()
        assert node.name == "test_node"

    def test_expected_parameters_exist(self) -> None:
        node = _make_node()
        param_names = [p.name for p in node.parameters]
        assert "config" in param_names
        assert "source_colorspace" in param_names
        assert "display" in param_names
        assert "view" in param_names
        assert "color_params" in param_names
        assert "was_successful" in param_names
        assert "result_details" in param_names

    def test_defaults_are_the_sentinel(self) -> None:
        # A freshly dropped node works the moment it is wired.
        node = _make_node()
        assert node.get_parameter_value("source_colorspace") == DEFAULT_SENTINEL
        assert node.get_parameter_value("display") == DEFAULT_SENTINEL
        assert node.get_parameter_value("view") == DEFAULT_SENTINEL

    def test_no_config_message_visible_by_default(self) -> None:
        node = _make_node()
        assert node._no_config_message.hide is False

    def test_dropdowns_visible_before_config_connected(self) -> None:
        node = _make_node()
        assert node._get_options_choices(node._source_colorspace_param) == [DEFAULT_SENTINEL]
        assert node._get_options_choices(node._display_param) == [DEFAULT_SENTINEL]
        assert node._get_options_choices(node._view_param) == [DEFAULT_SENTINEL]

    def test_no_config_message_hidden_on_reload_with_saved_config(self) -> None:
        mock_config = _mock_ocio_config()
        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node = OCIOColorParameters(
                "test_node",
                metadata={"_config_connected": True, "_config_file_path": None},
            )
        assert node._no_config_message.hide is True


class TestAfterValueSetConfig:
    def test_setting_config_populates_colorspace_choices(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        choices = node._get_options_choices(node._source_colorspace_param)
        assert "ACEScg" in choices
        assert "scene_linear" in choices

    def test_setting_config_populates_display_choices(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        choices = node._get_options_choices(node._display_param)
        assert "ACES" in choices
        assert "sRGB" in choices

    def test_setting_config_populates_view_choices_for_current_display(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()
        node.set_parameter_value("display", "ACES")

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        choices = node._get_options_choices(node._view_param)
        assert "sRGB" in choices
        assert "Log" in choices

    def test_setting_config_to_none_clears_choices(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        node.after_value_set(node._config_param, None)

        choices = node._get_options_choices(node._source_colorspace_param)
        assert "ACEScg" not in choices

    def test_update_options_writes_to_ui_options_not_just_trait(self) -> None:
        # Regression: choices must be written via param.update_ui_options_key so that
        # @emits_update_on_write fires and the canvas re-renders without squiggling.
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        # The canonical source of truth for the UI is ui_options["simple_dropdown"].
        assert "ACEScg" in node._source_colorspace_param.ui_options.get("simple_dropdown", [])
        assert "ACES" in node._display_param.ui_options.get("simple_dropdown", [])

    def test_no_config_message_hidden_when_config_set(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        assert node._no_config_message.hide is True

    def test_no_config_message_shown_when_config_cleared(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        node.after_value_set(node._config_param, None)

        assert node._no_config_message.hide is False

    def test_setting_config_includes_roles_in_colorspace_choices(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        choices = node._get_options_choices(node._source_colorspace_param)
        assert "scene_linear" in choices
        assert "compositing_log" in choices

    def test_roles_appear_before_colorspaces_in_choices(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        choices = node._get_options_choices(node._source_colorspace_param)
        role_idx = choices.index("scene_linear")
        cs_idx = choices.index("ACEScg")
        assert role_idx < cs_idx

    def test_config_saved_to_metadata_on_connect(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact(file_path="/path/to/config.ocio")
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        assert node.metadata.get("_config_connected") is True
        assert node.metadata.get("_config_file_path") == "/path/to/config.ocio"

    def test_config_env_var_mode_saved_to_metadata(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact(file_path=None)
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        assert node.metadata.get("_config_connected") is True
        assert node.metadata.get("_config_file_path") is None

    def test_config_cleared_from_metadata_on_disconnect(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        node.after_value_set(node._config_param, None)

        assert node.metadata.get("_config_connected") is False


class TestAfterValueSetDisplay:
    def test_changing_display_updates_view_choices(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)
            node.after_value_set(node._display_param, "sRGB")

        choices = node._get_options_choices(node._view_param)
        assert "Raw" in choices
        assert "Film" in choices
        assert "Log" not in choices  # ACES-specific view

    def test_changing_display_without_config_does_not_crash(self) -> None:
        node = _make_node()
        # No config connected — display change should be a no-op for choices
        node.after_value_set(node._display_param, "ACES")
        # Should not raise

    def test_changing_display_does_not_warn_view_for_display_name(self) -> None:
        # Regression: _validate_view was called with the display name instead of the
        # current view value, causing a spurious warning on every display change.
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)
            node.after_value_set(node._display_param, "ACES")

        # Cascade sets view to the first view for the selected display — no warning expected
        assert node._invalid_view_message.hide is True


class TestValidationMessages:
    def test_invalid_colorspace_shows_warning(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)

        node.set_parameter_value("source_colorspace", "NonExistentColorspace")
        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._source_colorspace_param, "NonExistentColorspace")

        assert node._invalid_colorspace_message.hide is False

    def test_valid_colorspace_hides_warning(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)
            node.set_parameter_value("source_colorspace", "ACEScg")
            node.after_value_set(node._source_colorspace_param, "ACEScg")

        assert node._invalid_colorspace_message.hide is True

    def test_invalid_display_shows_warning(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)
            node.after_value_set(node._display_param, "NonExistentDisplay")

        assert node._invalid_display_message.hide is False

    def test_valid_display_hides_warning(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, artifact)
            node.after_value_set(node._display_param, "ACES")

        assert node._invalid_display_message.hide is True


class TestReloadPersistence:
    def test_reload_with_saved_config_restores_choices(self) -> None:
        mock_config = _mock_ocio_config()
        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node = OCIOColorParameters(
                "test_node",
                metadata={"_config_connected": True, "_config_file_path": "/path/to/config.ocio"},
            )

        choices = node._get_options_choices(node._source_colorspace_param)
        assert "ACEScg" in choices

    def test_reload_without_saved_config_does_not_crash(self) -> None:
        node = OCIOColorParameters("test_node", metadata={})
        # No crash expected
        assert node.name == "test_node"

    def test_reload_with_env_var_config_restores_choices(self) -> None:
        mock_config = _mock_ocio_config()
        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node = OCIOColorParameters(
                "test_node",
                metadata={"_config_connected": True, "_config_file_path": None},
            )

        choices = node._get_options_choices(node._source_colorspace_param)
        assert "ACEScg" in choices


class TestProcess:
    def test_emits_artifact_without_config(self) -> None:
        node = _make_node()
        # All three are <default>, and with no config they resolve to empty,
        # which downstream transforms already treat as passthrough.
        node.process()

        artifact = node.parameter_output_values.get("color_params")
        assert isinstance(artifact, OCIOColorParamsArtifact)
        assert artifact.source_colorspace == ""
        assert artifact.display == ""
        assert artifact.view == ""

    def test_success_without_config(self) -> None:
        node = _make_node()
        node.process()
        assert node.parameter_output_values.get("was_successful") is True

    def test_emits_artifact_with_valid_config(self) -> None:
        node = _make_node()
        artifact_in = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            # Simulate framework propagation: calls after_value_set, which sets metadata.
            node.after_value_set(node._config_param, artifact_in)
            node.set_parameter_value("source_colorspace", "ACEScg")
            node.set_parameter_value("display", "ACES")
            node.set_parameter_value("view", "sRGB")
            node.process()

        artifact_out = node.parameter_output_values.get("color_params")
        assert isinstance(artifact_out, OCIOColorParamsArtifact)
        assert artifact_out.source_colorspace == "ACEScg"
        assert artifact_out.display == "ACES"
        assert artifact_out.view == "sRGB"
        assert node.parameter_output_values.get("was_successful") is True

    def test_config_with_none_file_path_uses_env_var(self) -> None:
        node = _make_node()
        artifact_in = _make_config_artifact(file_path=None)
        mock_config = _mock_ocio_config()

        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config
        ) as mock_load:
            # Simulate framework propagation: calls after_value_set, which sets metadata.
            node.after_value_set(node._config_param, artifact_in)
            node.set_parameter_value("source_colorspace", "ACEScg")
            node.process()

        # Validation in process() must call load_ocio_config(None) (triggers CreateFromEnv)
        mock_load.assert_called_with(None)

    def test_process_warns_when_selection_absent_from_execution_config(self) -> None:
        # Covers the safety-net path: user chose ACEScg from a full config's dropdown,
        # but at execution time the wired config no longer contains that colorspace.
        node = _make_node()
        artifact_in = _make_config_artifact()
        full_config = _mock_ocio_config()  # has ACEScg
        limited_config = _mock_ocio_config(
            colorspaces=["sRGB"],
            displays=["sRGB"],
            views_by_display={"sRGB": ["Raw"]},
            roles=[],
        )

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=full_config):
            node.after_value_set(node._config_param, artifact_in)  # populate dropdowns from full config
        node.set_parameter_value("source_colorspace", "ACEScg")  # valid choice in full config
        node.set_parameter_value("display", "ACES")

        # Metadata is already set from after_value_set above; process() uses it for validation.
        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=limited_config):
            node.process()

        result_details = node.parameter_output_values.get("result_details", "")
        assert "not found" in result_details
        assert node.parameter_output_values.get("was_successful") is True

    def test_process_forwards_explicit_config_path_to_artifact(self) -> None:
        node = _make_node()
        artifact_in = _make_config_artifact(file_path="/path/to/config.ocio")
        mock_config = _mock_ocio_config()

        with (
            patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config),
            patch(
                "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
                return_value=_resolved(config_path="/path/to/config.ocio"),
            ),
        ):
            # Simulate framework propagation: calls after_value_set, which sets metadata.
            node.after_value_set(node._config_param, artifact_in)
            node.process()

        artifact_out = node.parameter_output_values.get("color_params")
        assert isinstance(artifact_out, OCIOColorParamsArtifact)
        assert artifact_out.config_path == "/path/to/config.ocio"

    def test_process_forwards_none_config_path_for_env_var_mode(self) -> None:
        node = _make_node()
        artifact_in = _make_config_artifact(file_path=None)
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            # Simulate framework propagation: calls after_value_set, which sets metadata.
            node.after_value_set(node._config_param, artifact_in)
            node.process()

        artifact_out = node.parameter_output_values.get("color_params")
        assert isinstance(artifact_out, OCIOColorParamsArtifact)
        assert artifact_out.config_path is None

    def test_process_config_path_is_none_when_no_config_connected(self) -> None:
        node = _make_node()
        node.process()

        artifact_out = node.parameter_output_values.get("color_params")
        assert isinstance(artifact_out, OCIOColorParamsArtifact)
        assert artifact_out.config_path is None

    def test_process_role_alias_does_not_warn(self) -> None:
        node = _make_node()
        artifact_in = _make_config_artifact()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.set_parameter_value("config", artifact_in)
            node.set_parameter_value("source_colorspace", "scene_linear")  # role alias, not a colorspace name
            node.process()

        result_details = node.parameter_output_values.get("result_details", "")
        assert "not found" not in result_details
        assert node.parameter_output_values.get("was_successful") is True


def _make_connection_mocks(artifact) -> tuple:
    """Return (source_node, source_param) with artifact pre-loaded in output values."""
    source_node = MagicMock()
    source_param = MagicMock()
    source_param.name = "config"
    source_node.parameter_output_values = {"config": artifact}
    return source_node, source_param


class TestAfterIncomingConnection:
    def test_refreshes_dropdowns_immediately_when_upstream_has_value(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()
        source_node, source_param = _make_connection_mocks(artifact)

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_incoming_connection(source_node, source_param, node._config_param)

        choices = node._get_options_choices(node._display_param)
        assert "ACES" in choices
        assert "sRGB" in choices

    def test_source_colorspace_includes_roles_on_connection(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()
        source_node, source_param = _make_connection_mocks(artifact)

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_incoming_connection(source_node, source_param, node._config_param)

        choices = node._get_options_choices(node._source_colorspace_param)
        assert "scene_linear" in choices
        assert "ACEScg" in choices

    def test_hides_no_config_message_on_connection(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()
        source_node, source_param = _make_connection_mocks(artifact)

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_incoming_connection(source_node, source_param, node._config_param)

        assert node._no_config_message.hide is True

    def test_does_nothing_when_upstream_has_no_value(self) -> None:
        node = _make_node()
        source_node = MagicMock()
        source_param = MagicMock()
        source_param.name = "config"
        source_node.parameter_output_values = {}  # upstream hasn't run yet

        node.after_incoming_connection(source_node, source_param, node._config_param)

        # dropdowns should still show placeholder — no crash, no real choices
        assert node._get_options_choices(node._display_param) == [DEFAULT_SENTINEL]

    def test_ignores_connection_to_other_params(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()
        source_node, source_param = _make_connection_mocks(artifact)

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            # connect to display param, not config
            node.after_incoming_connection(source_node, source_param, node._display_param)

        # still placeholder choices only — config never connected via the hook
        assert node._get_options_choices(node._display_param) == [DEFAULT_SENTINEL]


class TestAfterIncomingConnectionRemoved:
    def test_clears_dropdowns_on_disconnect(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()
        source_node, source_param = _make_connection_mocks(artifact)

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_incoming_connection(source_node, source_param, node._config_param)
            node.after_incoming_connection_removed(source_node, source_param, node._config_param)

        assert node._get_options_choices(node._display_param) == [DEFAULT_SENTINEL]
        assert node._get_options_choices(node._source_colorspace_param) == [DEFAULT_SENTINEL]
        assert node._get_options_choices(node._view_param) == [DEFAULT_SENTINEL]

    def test_shows_no_config_message_on_disconnect(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()
        source_node, source_param = _make_connection_mocks(artifact)

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_incoming_connection(source_node, source_param, node._config_param)
            node.after_incoming_connection_removed(source_node, source_param, node._config_param)

        assert node._no_config_message.hide is False

    def test_clears_metadata_on_disconnect(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact(file_path="/some/config.ocio")
        mock_config = _mock_ocio_config()
        source_node, source_param = _make_connection_mocks(artifact)

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_incoming_connection(source_node, source_param, node._config_param)

        assert node.metadata.get("_config_connected") is True

        node.after_incoming_connection_removed(source_node, source_param, node._config_param)

        assert node.metadata.get("_config_connected") is False
        assert "_config_file_path" not in node.metadata

    def test_ignores_removal_for_other_params(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()
        source_node, source_param = _make_connection_mocks(artifact)

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_incoming_connection(source_node, source_param, node._config_param)
            # remove a connection from a different param — should not clear config dropdowns
            node.after_incoming_connection_removed(source_node, source_param, node._display_param)

        # choices should still be populated
        assert len(node._get_options_choices(node._display_param)) > 0


# ---------------------------------------------------------------------------
# <default> sentinel
# ---------------------------------------------------------------------------


def _resolved(**overrides: object) -> ResolvedColorParams:
    """Build a ResolvedColorParams with plausible values for patching the resolver."""
    defaults: dict[str, object] = {
        "working_space": "ACEScg",
        "source_colorspace": "ACEScg",
        "display": "sRGB - Display",
        "view": "ACES 1.0 - SDR Video",
        "config_path": None,
        "sources": {
            "working_space": "config-role",
            "source_colorspace": "config-role",
            "display": "config-default-display",
            "view": "config-default-view",
        },
        "warnings": (),
    }
    defaults.update(overrides)
    return ResolvedColorParams(**defaults)  # type: ignore[arg-type]


class TestSentinelChoices:
    def test_sentinel_is_the_first_choice_without_a_config(self) -> None:
        node = _make_node()
        for param in (node._source_colorspace_param, node._display_param, node._view_param):
            assert node._get_options_choices(param)[0] == DEFAULT_SENTINEL

    def test_sentinel_is_the_first_choice_with_a_config(self) -> None:
        node = _make_node()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, _make_config_artifact())

        for param in (node._source_colorspace_param, node._display_param, node._view_param):
            assert node._get_options_choices(param)[0] == DEFAULT_SENTINEL

    def test_real_choices_still_follow_the_sentinel(self) -> None:
        node = _make_node()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, _make_config_artifact())

        choices = node._get_options_choices(node._source_colorspace_param)
        assert "ACEScg" in choices
        assert "scene_linear" in choices

    def test_clearing_the_config_leaves_only_the_sentinel(self) -> None:
        node = _make_node()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, _make_config_artifact())
        node.after_value_set(node._config_param, None)

        assert node._get_options_choices(node._display_param) == [DEFAULT_SENTINEL]


class TestSentinelSurvivesOptionsTrait:
    def test_options_trait_allows_custom_values(self) -> None:
        # Without allow_custom the trait's converter rewrites any off-list value
        # to choices[0], which would mangle the sentinel on load.
        node = _make_node()
        for param in (node._source_colorspace_param, node._display_param, node._view_param):
            traits = list(param.find_elements_by_type(Options))
            assert traits
            assert traits[0].allow_custom is True

    def test_sentinel_survives_a_choices_refresh(self) -> None:
        node = _make_node()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, _make_config_artifact())

        assert node.get_parameter_value("source_colorspace") == DEFAULT_SENTINEL

    def test_an_explicit_value_survives_a_choices_refresh(self) -> None:
        node = _make_node()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, _make_config_artifact())
            node.set_parameter_value("source_colorspace", "ACEScg")
            node._refresh_all_dropdowns(None)

        assert node.get_parameter_value("source_colorspace") == "ACEScg"


class TestValidatorsExemptSentinel:
    def test_colorspace_validator_ignores_the_sentinel(self) -> None:
        node = _make_node()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, _make_config_artifact())
        node._validate_colorspace(DEFAULT_SENTINEL)

        assert node._invalid_colorspace_message.hide is True

    def test_display_validator_ignores_the_sentinel(self) -> None:
        node = _make_node()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, _make_config_artifact())
        node._validate_display(DEFAULT_SENTINEL)

        assert node._invalid_display_message.hide is True

    def test_view_validator_ignores_the_sentinel(self) -> None:
        node = _make_node()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, _make_config_artifact())
        node._validate_view(DEFAULT_SENTINEL)

        assert node._invalid_view_message.hide is True

    def test_an_unknown_value_is_still_reported(self) -> None:
        node = _make_node()
        mock_config = _mock_ocio_config()

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_value_set(node._config_param, _make_config_artifact())
        node._validate_colorspace("Not A Real Space")

        assert node._invalid_colorspace_message.hide is False


class TestBlankIsAnError:
    def test_a_blank_parameter_raises(self) -> None:
        node = _make_node()
        node.set_parameter_value("display", "")

        with pytest.raises(ValueError, match=DEFAULT_SENTINEL):
            node.process()

    def test_the_message_names_the_parameter(self) -> None:
        node = _make_node()
        node.set_parameter_value("view", "")

        with pytest.raises(ValueError, match="View"):
            node.process()

    def test_a_blank_parameter_is_not_migrated_to_the_sentinel(self) -> None:
        # Migrating silently would change rendered output.
        node = _make_node()
        node.set_parameter_value("display", "")

        with pytest.raises(ValueError, match=DEFAULT_SENTINEL):
            node.process()

        assert node.get_parameter_value("display") == ""

    def test_the_node_reports_failure(self) -> None:
        node = _make_node()
        node.set_parameter_value("display", "")

        with pytest.raises(ValueError, match=DEFAULT_SENTINEL):
            node.process()

        assert node.parameter_output_values.get("was_successful") is False


class TestSentinelResolvesOnEmit:
    def test_all_three_sentinels_resolve(self) -> None:
        node = _make_node()

        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
            return_value=_resolved(),
        ):
            node.process()

        artifact = node.parameter_output_values["color_params"]
        assert artifact.source_colorspace == "ACEScg"
        assert artifact.display == "sRGB - Display"
        assert artifact.view == "ACES 1.0 - SDR Video"

    def test_one_sentinel_resolves_while_the_others_stay_explicit(self) -> None:
        node = _make_node()
        node.set_parameter_value("display", "Rec.1886 Rec.709 - Display")
        node.set_parameter_value("view", "Video")

        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
            return_value=_resolved(
                source_colorspace="ACEScg",
                display="Rec.1886 Rec.709 - Display",
                view="Video",
            ),
        ) as resolver:
            node.process()

        # The explicit values are handed to the resolver, which returns them unchanged.
        explicit = resolver.call_args.kwargs["explicit"]
        assert explicit.display == "Rec.1886 Rec.709 - Display"
        assert explicit.view == "Video"
        assert explicit.source_colorspace == ""

    def test_a_fully_explicit_node_does_not_call_the_resolver(self) -> None:
        node = _make_node()
        node.set_parameter_value("source_colorspace", "ACEScg")
        node.set_parameter_value("display", "sRGB")
        node.set_parameter_value("view", "Film")

        with patch("opencolorio.nodes.config.ocio_color_parameters.resolve_color_params") as resolver:
            node.process()

        resolver.assert_not_called()

    def test_a_fully_explicit_node_emits_its_own_values(self) -> None:
        node = _make_node()
        node.set_parameter_value("source_colorspace", "ACEScg")
        node.set_parameter_value("display", "sRGB")
        node.set_parameter_value("view", "Film")

        node.process()

        artifact = node.parameter_output_values["color_params"]
        assert artifact.source_colorspace == "ACEScg"
        assert artifact.display == "sRGB"

    def test_resolver_warnings_reach_result_details(self) -> None:
        node = _make_node()

        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
            return_value=_resolved(warnings=("something is display-referred",)),
        ):
            node.process()

        assert "display-referred" in str(node.parameter_output_values.get("result_details"))

    def test_provenance_reaches_result_details(self) -> None:
        node = _make_node()

        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
            return_value=_resolved(),
        ):
            node.process()

        assert "config-default-display" in str(node.parameter_output_values.get("result_details"))


class TestDefaultHints:
    def test_the_resolved_value_appears_as_a_placeholder(self) -> None:
        node = _make_node()

        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
            return_value=_resolved(),
        ):
            node._refresh_default_hints()

        assert node._display_param.ui_options.get("placeholder_text") == "sRGB - Display"

    def test_the_placeholder_does_not_become_the_value(self) -> None:
        node = _make_node()

        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
            return_value=_resolved(),
        ):
            node._refresh_default_hints()

        assert node.get_parameter_value("display") == DEFAULT_SENTINEL

    def test_the_badge_names_the_layer(self) -> None:
        node = _make_node()

        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
            return_value=_resolved(),
        ):
            node._refresh_default_hints()

        badge = node._display_param.ui_options.get("badge")
        assert badge is not None
        assert "default display" in badge["message"]

    def test_the_badge_names_the_config(self) -> None:
        node = _make_node()

        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
            return_value=_resolved(config_path="/studio/show.ocio"),
        ):
            node._refresh_default_hints()

        badge = node._display_param.ui_options.get("badge")
        assert badge is not None
        assert "/studio/show.ocio" in badge["message"]

    def test_an_explicit_parameter_gets_no_hint(self) -> None:
        node = _make_node()
        node.set_parameter_value("display", "sRGB")

        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
            return_value=_resolved(),
        ):
            node._refresh_default_hints()

        assert node._display_param.ui_options.get("placeholder_text") in (None, "")
        assert node._display_param.ui_options.get("badge") is None

    def test_a_failing_resolver_does_not_break_the_node(self) -> None:
        node = _make_node()

        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
            side_effect=RuntimeError("config exploded"),
        ):
            node._refresh_default_hints()  # must not raise

    def test_node_construction_survives_a_failing_resolver(self) -> None:
        with patch(
            "opencolorio.nodes.config.ocio_color_parameters.resolve_color_params",
            side_effect=RuntimeError("config exploded"),
        ):
            assert OCIOColorParameters("test_node") is not None
