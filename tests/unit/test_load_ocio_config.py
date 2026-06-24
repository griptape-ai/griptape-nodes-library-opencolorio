from __future__ import annotations

from unittest.mock import MagicMock, patch

import PyOpenColorIO as ocio
import pytest
from griptape_nodes.files.file import FileLoadError
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

from opencolorio.artifacts.ocio_config_artifact import OCIOConfigArtifact
from opencolorio.nodes.config.load_ocio_config import LoadOCIOConfig
from opencolorio.ocio_helpers import extract_lists, load_ocio_config

# ---------------------------------------------------------------------------
# Module-level helper tests — no node instantiation needed
# ---------------------------------------------------------------------------


class TestLoadOCIOConfigHelper:
    def test_empty_path_calls_create_from_env(self) -> None:
        mock_config = MagicMock()
        with patch("opencolorio.ocio_helpers.ocio") as mock_ocio:
            mock_ocio.Config.CreateFromEnv.return_value = mock_config
            result = load_ocio_config("")
        mock_ocio.Config.CreateFromEnv.assert_called_once()
        assert result is mock_config

    def test_none_path_calls_create_from_env(self) -> None:
        mock_config = MagicMock()
        with patch("opencolorio.ocio_helpers.ocio") as mock_ocio:
            mock_ocio.Config.CreateFromEnv.return_value = mock_config
            result = load_ocio_config(None)
        mock_ocio.Config.CreateFromEnv.assert_called_once()
        assert result is mock_config

    def test_non_empty_path_calls_create_from_file(self) -> None:
        mock_config = MagicMock()
        with patch("opencolorio.ocio_helpers.ocio") as mock_ocio:
            mock_ocio.Config.CreateFromFile.return_value = mock_config
            result = load_ocio_config("/some/config.ocio")
        mock_ocio.Config.CreateFromFile.assert_called_once_with("/some/config.ocio")
        assert result is mock_config


class TestExtractLists:
    def test_returns_three_lists_of_strings(self) -> None:
        # CreateRaw() gives a minimal valid config with no colorspaces/displays/roles —
        # enough to verify the return types without a real .ocio file on disk.
        config = ocio.Config.CreateRaw()
        colorspaces, displays, roles = extract_lists(config)
        assert isinstance(colorspaces, list)
        assert isinstance(displays, list)
        assert isinstance(roles, list)
        assert all(isinstance(cs, str) for cs in colorspaces)
        assert all(isinstance(d, str) for d in displays)
        assert all(isinstance(r, str) for r in roles)


# ---------------------------------------------------------------------------
# Node tests — GriptapeNodes is mocked via conftest autouse fixture
# ---------------------------------------------------------------------------


def _make_node() -> LoadOCIOConfig:
    return LoadOCIOConfig("test_node")


def _make_file_load_error() -> FileLoadError:
    return FileLoadError(FileIOFailureReason.MISSING_MACRO_VARIABLES, "unresolved macro variable")


class TestResolvePath:
    def test_empty_string_returns_none(self) -> None:
        node = _make_node()
        assert node._resolve_path("") is None

    def test_valid_path_returns_resolved(self) -> None:
        node = _make_node()
        with patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls:
            mock_file_cls.return_value.resolve.return_value = "/abs/resolved/config.ocio"
            result = node._resolve_path("/some/path.ocio")
        assert result == "/abs/resolved/config.ocio"

    def test_file_load_error_propagates(self) -> None:
        node = _make_node()
        with patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls:
            mock_file_cls.return_value.resolve.side_effect = _make_file_load_error()
            with pytest.raises(FileLoadError):
                node._resolve_path("{UNDEFINED}/config.ocio")


class TestNodeInstantiation:
    def test_node_can_be_created(self) -> None:
        node = _make_node()
        assert node.name == "test_node"

    def test_ocio_changed_message_hidden_when_override_was_active(self) -> None:
        # $OCIO changed since last run, but last run used override → no staleness warning.
        node = LoadOCIOConfig(
            "test_node",
            metadata={"_override_enabled": True, "_saved_ocio_env": "/old.ocio"},
        )
        assert node._ocio_changed_message.hide is True

    def test_expected_parameters_exist(self) -> None:
        node = _make_node()
        param_names = [p.name for p in node.parameters]
        assert "file_path" in param_names
        assert "context_vars" in param_names
        assert "config" in param_names
        assert "was_successful" in param_names
        assert "result_details" in param_names

    def test_list_outputs_removed(self) -> None:
        node = _make_node()
        param_names = [p.name for p in node.parameters]
        assert "colorspace_names" not in param_names
        assert "display_names" not in param_names
        assert "role_names" not in param_names


class TestAfterValueSet:
    def test_toggle_clears_execution_status(self) -> None:
        node = _make_node()
        node._set_status_results(was_successful=True, result_details="Loaded from $OCIO")
        node.after_value_set(node._override_param, True)
        assert node.parameter_output_values.get("was_successful") is not True

    def test_file_path_change_clears_status_when_override_enabled(self) -> None:
        node = _make_node()
        node._set_status_results(was_successful=True, result_details="Old result")
        node.set_parameter_value("override_ocio_config", True)
        with patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls:
            mock_file_cls.return_value.resolve.return_value = "/abs/config.ocio"
            node.after_value_set(node._file_path_param, "/abs/config.ocio")
        assert node.parameter_output_values.get("was_successful") is not True

    def test_file_path_ignored_when_override_disabled(self) -> None:
        node = _make_node()
        # Override is OFF by default; setting file_path should not load the config.
        with patch("opencolorio.nodes.config.load_ocio_config.load_ocio_config") as mock_load:
            node.after_value_set(node._file_path_param, "/abs/config.ocio")
            mock_load.assert_not_called()

    def test_file_load_error_clears_config_param(self) -> None:
        node = _make_node()
        # Seed a prior successful artifact so we can verify it's cleared.
        node.parameter_output_values["config"] = OCIOConfigArtifact(file_path="/old/config.ocio")

        with patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls:
            mock_file_cls.return_value.resolve.side_effect = _make_file_load_error()
            node.after_value_set(node._file_path_param, "{BAD_VAR}/config.ocio")

        assert node.parameter_output_values.get("config") is None

    def test_file_load_error_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        node = _make_node()
        with (
            patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls,
            caplog.at_level(logging.WARNING, logger="griptape_nodes"),
        ):
            mock_file_cls.return_value.resolve.side_effect = _make_file_load_error()
            node.after_value_set(node._file_path_param, "{BAD}/config.ocio")

        assert any("could not resolve path" in r.message for r in caplog.records)


class TestProcess:
    def test_success_uses_env_var_by_default(self) -> None:
        node = _make_node()
        raw_config = ocio.Config.CreateRaw()

        with patch("opencolorio.nodes.config.load_ocio_config.load_ocio_config", return_value=raw_config):
            node.process()

        artifact = node.parameter_output_values.get("config")
        assert isinstance(artifact, OCIOConfigArtifact)
        assert artifact.file_path is None  # env-var mode: None signals CreateFromEnv()

    def test_success_uses_override_path_when_enabled(self) -> None:
        node = _make_node()
        raw_config = ocio.Config.CreateRaw()
        node.set_parameter_value("override_ocio_config", True)

        with (
            patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls,
            patch("opencolorio.nodes.config.load_ocio_config.load_ocio_config", return_value=raw_config),
        ):
            mock_file_cls.return_value.resolve.return_value = "/abs/config.ocio"
            node.set_parameter_value("file_path", "/abs/config.ocio")
            node.process()

        artifact = node.parameter_output_values.get("config")
        assert isinstance(artifact, OCIOConfigArtifact)
        assert artifact.file_path == "/abs/config.ocio"

    def test_success_sets_was_successful_true(self) -> None:
        node = _make_node()
        raw_config = ocio.Config.CreateRaw()

        with (
            patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls,
            patch("opencolorio.nodes.config.load_ocio_config.load_ocio_config", return_value=raw_config),
        ):
            mock_file_cls.return_value.resolve.return_value = "/abs/config.ocio"
            node.set_parameter_value("file_path", "/abs/config.ocio")
            node.process()

        assert node.parameter_output_values.get("was_successful") is True

    def test_failure_sets_was_successful_false(self) -> None:
        node = _make_node()

        with patch("opencolorio.nodes.config.load_ocio_config.load_ocio_config") as mock_load:
            mock_load.side_effect = Exception("bad config")
            with pytest.raises(Exception, match="bad config"):
                node.process()

        assert node.parameter_output_values.get("was_successful") is False

    def test_failure_includes_error_in_result_details(self) -> None:
        node = _make_node()

        with patch("opencolorio.nodes.config.load_ocio_config.load_ocio_config") as mock_load:
            mock_load.side_effect = RuntimeError("config file not found")
            with pytest.raises(RuntimeError):
                node.process()

        details = node.parameter_output_values.get("result_details", "")
        assert "config file not found" in details

    def test_override_with_invalid_path_sets_failure(self) -> None:
        # Override ON + bad path must fail visibly, not silently fall back to $OCIO.
        node = _make_node()
        node.set_parameter_value("override_ocio_config", True)
        with patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls:
            mock_file_cls.return_value.resolve.side_effect = _make_file_load_error()
            node.set_parameter_value("file_path", "{BAD}/config.ocio")
            with pytest.raises(FileLoadError):
                node.process()
        assert node.parameter_output_values.get("was_successful") is False

    def test_override_run_clears_saved_ocio_env(self) -> None:
        # Running in override mode must clear the saved env-var to prevent stale staleness warnings.
        node = _make_node()
        node.metadata["_saved_ocio_env"] = "/old/env.ocio"
        node.set_parameter_value("override_ocio_config", True)
        raw_config = ocio.Config.CreateRaw()
        with (
            patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls,
            patch("opencolorio.nodes.config.load_ocio_config.load_ocio_config", return_value=raw_config),
        ):
            mock_file_cls.return_value.resolve.return_value = "/abs/config.ocio"
            node.set_parameter_value("file_path", "/abs/config.ocio")
            node.process()
        assert "_saved_ocio_env" not in node.metadata

    def test_env_var_run_saves_ocio_env(self) -> None:
        node = _make_node()
        raw_config = ocio.Config.CreateRaw()
        with (
            patch("opencolorio.nodes.config.load_ocio_config.load_ocio_config", return_value=raw_config),
            patch.dict("os.environ", {"OCIO": "/studio/config.ocio"}),
        ):
            node.process()
        assert node.metadata.get("_saved_ocio_env") == "/studio/config.ocio"
