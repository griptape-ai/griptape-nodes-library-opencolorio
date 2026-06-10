from __future__ import annotations

from unittest.mock import MagicMock, patch

import PyOpenColorIO as ocio
import pytest
from griptape_nodes.files.file import FileLoadError
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

from opencolorio.artifacts.ocio_config_artifact import OCIOConfigArtifact
from opencolorio.nodes.config.load_ocio_config import LoadOCIOConfig, _extract_lists, _load_ocio_config

# ---------------------------------------------------------------------------
# Module-level helper tests — no node instantiation needed
# ---------------------------------------------------------------------------


class TestLoadOCIOConfigHelper:
    def test_empty_path_calls_create_from_env(self) -> None:
        mock_config = MagicMock()
        with patch("opencolorio.nodes.config.load_ocio_config.ocio") as mock_ocio:
            mock_ocio.Config.CreateFromEnv.return_value = mock_config
            result = _load_ocio_config("")
        mock_ocio.Config.CreateFromEnv.assert_called_once()
        assert result is mock_config

    def test_non_empty_path_calls_create_from_file(self) -> None:
        mock_config = MagicMock()
        with patch("opencolorio.nodes.config.load_ocio_config.ocio") as mock_ocio:
            mock_ocio.Config.CreateFromFile.return_value = mock_config
            result = _load_ocio_config("/some/config.ocio")
        mock_ocio.Config.CreateFromFile.assert_called_once_with("/some/config.ocio")
        assert result is mock_config


class TestExtractLists:
    def test_returns_three_lists_of_strings(self) -> None:
        # CreateRaw() gives a minimal valid config with no colorspaces/displays/roles —
        # enough to verify the return types without a real .ocio file on disk.
        config = ocio.Config.CreateRaw()
        colorspaces, displays, roles = _extract_lists(config)
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
    def test_empty_string_returns_empty(self) -> None:
        node = _make_node()
        assert node._resolve_path("") == ""

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

    def test_expected_parameters_exist(self) -> None:
        node = _make_node()
        param_names = [p.name for p in node.parameters]
        assert "file_path" in param_names
        assert "context_vars" in param_names
        assert "config" in param_names
        assert "colorspace_names" in param_names
        assert "display_names" in param_names
        assert "role_names" in param_names
        assert "was_successful" in param_names
        assert "result_details" in param_names


class TestAfterValueSet:
    def test_valid_path_populates_lists(self) -> None:
        node = _make_node()
        raw_config = ocio.Config.CreateRaw()

        with (
            patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls,
            patch("opencolorio.nodes.config.load_ocio_config._load_ocio_config", return_value=raw_config),
        ):
            mock_file_cls.return_value.resolve.return_value = "/abs/config.ocio"
            node.after_value_set(node._file_path_param, "/abs/config.ocio")

        assert isinstance(node.parameter_output_values.get("colorspace_names"), list)
        assert isinstance(node.parameter_output_values.get("display_names"), list)
        assert isinstance(node.parameter_output_values.get("role_names"), list)

    def test_file_load_error_clears_lists(self) -> None:
        node = _make_node()
        # Seed some values so we can verify they're cleared.
        node.parameter_output_values["colorspace_names"] = ["sRGB"]
        node.parameter_output_values["display_names"] = ["ACES"]
        node.parameter_output_values["role_names"] = ["scene_linear"]

        with patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls:
            mock_file_cls.return_value.resolve.side_effect = _make_file_load_error()
            node.after_value_set(node._file_path_param, "{BAD_VAR}/config.ocio")

        assert node.parameter_output_values["colorspace_names"] == []
        assert node.parameter_output_values["display_names"] == []
        assert node.parameter_output_values["role_names"] == []

    def test_file_load_error_stores_empty_metadata_path(self) -> None:
        node = _make_node()
        with patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls:
            mock_file_cls.return_value.resolve.side_effect = _make_file_load_error()
            node.after_value_set(node._file_path_param, "{BAD}/config.ocio")

        assert node.metadata.get("_file_path") == ""

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


class TestAprocess:
    async def test_success_sets_config_artifact(self) -> None:
        node = _make_node()
        raw_config = ocio.Config.CreateRaw()

        with (
            patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls,
            patch("opencolorio.nodes.config.load_ocio_config._load_ocio_config", return_value=raw_config),
        ):
            mock_file_cls.return_value.resolve.return_value = "/abs/config.ocio"
            node.set_parameter_value("file_path", "/abs/config.ocio")
            await node.aprocess()

        artifact = node.parameter_output_values.get("config")
        assert isinstance(artifact, OCIOConfigArtifact)
        assert artifact.file_path == "/abs/config.ocio"

    async def test_success_sets_was_successful_true(self) -> None:
        node = _make_node()
        raw_config = ocio.Config.CreateRaw()

        with (
            patch("opencolorio.nodes.config.load_ocio_config.File") as mock_file_cls,
            patch("opencolorio.nodes.config.load_ocio_config._load_ocio_config", return_value=raw_config),
        ):
            mock_file_cls.return_value.resolve.return_value = "/abs/config.ocio"
            node.set_parameter_value("file_path", "/abs/config.ocio")
            await node.aprocess()

        assert node.parameter_output_values.get("was_successful") is True

    async def test_failure_sets_was_successful_false(self) -> None:
        node = _make_node()

        with (
            patch("opencolorio.nodes.config.load_ocio_config._load_ocio_config") as mock_load,
        ):
            mock_load.side_effect = Exception("bad config")
            with pytest.raises(Exception, match="bad config"):
                await node.aprocess()

        assert node.parameter_output_values.get("was_successful") is False

    async def test_failure_includes_error_in_result_details(self) -> None:
        node = _make_node()

        with patch("opencolorio.nodes.config.load_ocio_config._load_ocio_config") as mock_load:
            mock_load.side_effect = RuntimeError("config file not found")
            with pytest.raises(RuntimeError):
                await node.aprocess()

        details = node.parameter_output_values.get("result_details", "")
        assert "config file not found" in details
