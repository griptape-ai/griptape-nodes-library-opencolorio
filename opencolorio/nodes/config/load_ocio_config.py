from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.file_system_picker import FileSystemPicker

from opencolorio.artifacts.ocio_config_artifact import OCIOConfigArtifact
from opencolorio.ocio_helpers import extract_lists, load_ocio_config

logger = logging.getLogger("griptape_nodes")


class LoadOCIOConfig(SuccessFailureNode):
    """Load an OpenColorIO config from a file path or the $OCIO environment variable.

    Setting the file path populates colorspace, display, and role lists immediately
    for use in downstream node dropdowns. Executing the node emits an
    OCIOConfigArtifact that carries the path and any context variables forward.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self._file_path_param = ParameterString(
            name="file_path",
            display_name="File Path",
            default_value="",
            tooltip="Path to an OCIO config file. Leave empty to load from the $OCIO environment variable.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self._file_path_param.add_trait(
            FileSystemPicker(
                allow_files=True,
                allow_directories=False,
                file_extensions=[".ocio"],
            )
        )
        self.add_parameter(self._file_path_param)

        self._context_vars_param = Parameter(
            name="context_vars",
            display_name="Context Variables",
            type="dict",
            default_value={},
            tooltip='OCIO context variables, e.g. {"SHOT": "sh010", "SEQ": "sq020"}',
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(self._context_vars_param)

        self._config_param = Parameter(
            name="config",
            display_name="Config",
            type="OCIOConfigArtifact",
            output_type="OCIOConfigArtifact",
            tooltip="OCIOConfigArtifact carrying the resolved config path and context variables",
            allowed_modes={ParameterMode.OUTPUT},
            settable=False,
        )
        self.add_parameter(self._config_param)

        self._colorspace_names_param = Parameter(
            name="colorspace_names",
            display_name="Colorspace Names",
            type="list[str]",
            output_type="list[str]",
            tooltip="All colorspace names defined in the config",
            allowed_modes={ParameterMode.OUTPUT},
            settable=False,
        )
        self.add_parameter(self._colorspace_names_param)

        self._display_names_param = Parameter(
            name="display_names",
            display_name="Display Names",
            type="list[str]",
            output_type="list[str]",
            tooltip="All display names defined in the config",
            allowed_modes={ParameterMode.OUTPUT},
            settable=False,
        )
        self.add_parameter(self._display_names_param)

        self._role_names_param = Parameter(
            name="role_names",
            display_name="Role Names",
            type="list[str]",
            output_type="list[str]",
            tooltip='Stable role aliases defined in the config (e.g. "scene_linear", "compositing_log")',
            allowed_modes={ParameterMode.OUTPUT},
            settable=False,
        )
        self.add_parameter(self._role_names_param)

        self._create_status_parameters(
            result_details_tooltip="Details about the OCIO config load result",
            result_details_placeholder="Config details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        # Restore list outputs if saved with a valid path.
        saved_path: str | None = self.metadata.get("_file_path")
        if saved_path:
            self._refresh_lists(saved_path)

    # --- Lifecycle ---

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter is self._file_path_param:
            raw = str(value) if value else None
            try:
                resolved = self._resolve_path(raw)
            except FileLoadError as e:
                logger.warning("LoadOCIOConfig '%s': could not resolve path '%s': %s", self.name, raw, e)
                self.metadata["_file_path"] = None
                self.parameter_output_values[self._config_param.name] = None
                self._set_list_outputs([], [], [])
                return super().after_value_set(parameter, value)
            self.metadata["_file_path"] = resolved
            self._refresh_lists(resolved)
        return super().after_value_set(parameter, value)

    async def aprocess(self) -> None:
        self._clear_execution_status()

        try:
            raw = self.get_parameter_value(self._file_path_param.name) or None
            file_path = self._resolve_path(raw)
            context_vars = self.get_parameter_value(self._context_vars_param.name) or {}

            config = load_ocio_config(file_path)
            colorspace_names, display_names, role_names = extract_lists(config)

            self.parameter_output_values[self._config_param.name] = OCIOConfigArtifact(
                file_path=file_path,
                context_vars=dict(context_vars),
            )
            self._set_list_outputs(colorspace_names, display_names, role_names)

            src = file_path or "$OCIO"
            self._set_status_results(
                was_successful=True,
                result_details=(
                    f"Loaded from {src} — "
                    f"{len(colorspace_names)} colorspaces, "
                    f"{len(display_names)} displays, "
                    f"{len(role_names)} roles"
                ),
            )

        except Exception as e:
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)

    # --- Private ---

    def _resolve_path(self, raw: str | None) -> str | None:
        if not raw:
            return None
        return File(raw).resolve()

    def _refresh_lists(self, file_path: str | None) -> None:
        """Load config and populate list outputs. Clears lists silently on failure."""
        try:
            config = load_ocio_config(file_path)
            colorspace_names, display_names, role_names = extract_lists(config)
        except Exception as e:
            logger.warning(
                "LoadOCIOConfig '%s': could not load config from '%s': %s", self.name, file_path or "$OCIO", e
            )
            colorspace_names, display_names, role_names = [], [], []

        self._set_list_outputs(colorspace_names, display_names, role_names)

    def _set_list_outputs(
        self,
        colorspace_names: list[str],
        display_names: list[str],
        role_names: list[str],
    ) -> None:
        self.parameter_output_values[self._colorspace_names_param.name] = colorspace_names
        self.parameter_output_values[self._display_names_param.name] = display_names
        self.parameter_output_values[self._role_names_param.name] = role_names
