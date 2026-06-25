from __future__ import annotations

from unittest.mock import MagicMock, patch

from opencolorio.artifacts.ocio_color_params_artifact import OCIOColorParamsArtifact
from opencolorio.artifacts.ocio_config_artifact import OCIOConfigArtifact
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

    def test_defaults_are_empty_strings(self) -> None:
        node = _make_node()
        assert node.get_parameter_value("source_colorspace") == ""
        assert node.get_parameter_value("display") == ""
        assert node.get_parameter_value("view") == ""

    def test_no_config_message_visible_by_default(self) -> None:
        node = _make_node()
        assert node._no_config_message.hide is False

    def test_dropdowns_visible_before_config_connected(self) -> None:
        node = _make_node()
        assert node._get_options_choices(node._source_colorspace_param) == [""]
        assert node._get_options_choices(node._display_param) == [""]
        assert node._get_options_choices(node._view_param) == [""]

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
        # Without a config connected, all params are locked to placeholder "".
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
            # Connect config first so choices are populated before we select values.
            node.set_parameter_value("config", artifact_in)
            node.set_parameter_value("source_colorspace", "ACEScg")
            node.set_parameter_value("display", "ACES")
            node.set_parameter_value("view", "sRGB")
            # Simulate framework propagating the wired value to parameter_output_values.
            node.parameter_output_values["config"] = artifact_in
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
            node.set_parameter_value("config", artifact_in)
            node.set_parameter_value("source_colorspace", "ACEScg")
            # Simulate framework propagating the wired value to parameter_output_values.
            node.parameter_output_values["config"] = artifact_in
            node.process()

        # Must pass None (triggers CreateFromEnv) not raise
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

        # At execution time the wired config has changed (simulate via parameter_output_values).
        node.parameter_output_values["config"] = artifact_in
        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=limited_config):
            node.process()

        result_details = node.parameter_output_values.get("result_details", "")
        assert "not found" in result_details
        assert node.parameter_output_values.get("was_successful") is True

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
        assert node._get_options_choices(node._display_param) == [""]

    def test_ignores_connection_to_other_params(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()
        source_node, source_param = _make_connection_mocks(artifact)

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            # connect to display param, not config
            node.after_incoming_connection(source_node, source_param, node._display_param)

        # still placeholder choices only — config never connected via the hook
        assert node._get_options_choices(node._display_param) == [""]


class TestAfterIncomingConnectionRemoved:
    def test_clears_dropdowns_on_disconnect(self) -> None:
        node = _make_node()
        artifact = _make_config_artifact()
        mock_config = _mock_ocio_config()
        source_node, source_param = _make_connection_mocks(artifact)

        with patch("opencolorio.nodes.config.ocio_color_parameters.load_ocio_config", return_value=mock_config):
            node.after_incoming_connection(source_node, source_param, node._config_param)
            node.after_incoming_connection_removed(source_node, source_param, node._config_param)

        assert node._get_options_choices(node._display_param) == [""]
        assert node._get_options_choices(node._source_colorspace_param) == [""]
        assert node._get_options_choices(node._view_param) == [""]

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
