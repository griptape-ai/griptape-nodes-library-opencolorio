from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

from opencolorio.artifacts.ocio_color_params_artifact import OCIOColorParamsArtifact
from opencolorio.artifacts.ocio_config_artifact import OCIOConfigArtifact
from opencolorio.ocio_helpers import extract_lists, load_ocio_config

logger = logging.getLogger("griptape_nodes")


class OCIOColorParameters(SuccessFailureNode):
    """Bundle source colorspace, display, and view into a reusable OCIOColorParamsArtifact.

    When an OCIOConfigArtifact is wired to the ``config`` input the three
    dropdowns are populated from that config and restricted to valid combinations.
    Without a config, dropdowns show a placeholder and emit empty-string values.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # --- Optional config input ---
        self._config_param = Parameter(
            name="config",
            display_name="OCIO Config",
            type="OCIOConfigArtifact",
            tooltip="Optional OCIOConfigArtifact from LoadOCIOConfig — enables dropdown population and validation",
            allowed_modes={ParameterMode.INPUT},
            settable=False,
        )
        self.add_parameter(self._config_param)

        self._no_config_message = ParameterMessage(
            name="no_config_message",
            variant="info",
            value="Connect an **OCIO Config** to enable colorspace dropdowns.",
            markdown=True,
            hide=self.metadata.get("_config_connected", False),
        )
        self.add_node_element(self._no_config_message)

        # Standard Options trait with [""] placeholder so the dropdown widget always renders.
        # _update_option_choices (inherited from BaseNode) updates choices and triggers UI refresh.

        self._source_colorspace_param = ParameterString(
            name="source_colorspace",
            display_name="Source Colorspace",
            default_value="",
            tooltip="Source colorspace name selected from the connected OCIO config",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[""])},
        )
        self.add_parameter(self._source_colorspace_param)

        self._display_param = ParameterString(
            name="display",
            display_name="Display",
            default_value="",
            tooltip="Display device name selected from the connected OCIO config",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[""])},
        )
        self.add_parameter(self._display_param)

        self._view_param = ParameterString(
            name="view",
            display_name="View",
            default_value="",
            tooltip="View name for the selected display",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[""])},
        )
        self.add_parameter(self._view_param)

        # --- Validation messages (hidden by default) ---
        self._invalid_colorspace_message = ParameterMessage(
            name="invalid_colorspace_message",
            variant="warning",
            value="",
            markdown=True,
            hide=True,
        )
        self.add_node_element(self._invalid_colorspace_message)

        self._invalid_display_message = ParameterMessage(
            name="invalid_display_message",
            variant="warning",
            value="",
            markdown=True,
            hide=True,
        )
        self.add_node_element(self._invalid_display_message)

        self._invalid_view_message = ParameterMessage(
            name="invalid_view_message",
            variant="warning",
            value="",
            markdown=True,
            hide=True,
        )
        self.add_node_element(self._invalid_view_message)

        # --- Output ---
        self._color_params_output = Parameter(
            name="color_params",
            display_name="Color Parameters",
            type="OCIOColorParamsArtifact",
            output_type="OCIOColorParamsArtifact",
            tooltip="OCIOColorParamsArtifact carrying source_colorspace, display, and view",
            allowed_modes={ParameterMode.OUTPUT},
            settable=False,
        )
        self.add_parameter(self._color_params_output)

        self._create_status_parameters(
            result_details_tooltip="Validation and emit details for the color parameter bundle",
            result_details_placeholder="Color parameters will appear here.",
            parameter_group_initially_collapsed=True,
        )

        # Restore dropdowns on reload (after_value_set is skipped during restore).
        if self.metadata.get("_config_connected"):
            saved_file_path = self.metadata.get("_config_file_path")  # may be None (env-var mode)
            self._refresh_all_dropdowns(saved_file_path)

    # --- Lifecycle ---

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter is self._config_param:
            if value is not None and isinstance(value, OCIOConfigArtifact):
                self.metadata["_config_connected"] = True
                self.metadata["_config_file_path"] = value.file_path
                self._refresh_all_dropdowns(value.file_path)
                self.hide_message_by_name("no_config_message")
            else:
                self.metadata["_config_connected"] = False
                self.metadata.pop("_config_file_path", None)
                self._clear_dropdowns()
                self.show_message_by_name("no_config_message")

        elif parameter is self._display_param:
            if self.metadata.get("_config_connected"):
                file_path = self.metadata.get("_config_file_path")
                self._refresh_view_choices(file_path, str(value) if value else "")
                self._validate_display(str(value) if value else "")
                self._validate_view(self.get_parameter_value("view") or "")

        elif parameter is self._source_colorspace_param:
            if self.metadata.get("_config_connected"):
                self._validate_colorspace(str(value) if value else "")

        elif parameter is self._view_param:
            if self.metadata.get("_config_connected"):
                self._validate_view(str(value) if value else "")

        return super().after_value_set(parameter, value)

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        if target_parameter is self._config_param:
            artifact = source_node.parameter_output_values.get(source_parameter.name)
            if isinstance(artifact, OCIOConfigArtifact):
                self.metadata["_config_connected"] = True
                self.metadata["_config_file_path"] = artifact.file_path
                self._refresh_all_dropdowns(artifact.file_path)
                self.hide_message_by_name("no_config_message")
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        if target_parameter is self._config_param:
            self.metadata["_config_connected"] = False
            self.metadata.pop("_config_file_path", None)
            self._clear_dropdowns()
            self.show_message_by_name("no_config_message")
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def process(self) -> None:
        self._clear_execution_status()

        source_colorspace = self.get_parameter_value("source_colorspace") or ""
        display = self.get_parameter_value("display") or ""
        view = self.get_parameter_value("view") or ""

        # Read config state from metadata — set by after_value_set when the framework
        # propagates the upstream value. parameter_output_values is cleared before
        # execution so cannot be used here.
        config_connected: bool = self.metadata.get("_config_connected", False)
        config_file_path: str | None = self.metadata.get("_config_file_path") if config_connected else None
        warnings: list[str] = []

        if config_connected:
            try:
                config = load_ocio_config(config_file_path)
                colorspace_names = list(config.getColorSpaceNames())
                role_names = [r for r, _ in config.getRoles()]
                display_names = list(config.getDisplays())
                view_names = list(config.getViews(display)) if display else []

                if (
                    source_colorspace
                    and source_colorspace not in colorspace_names
                    and source_colorspace not in role_names
                ):
                    warnings.append(f"source_colorspace '{source_colorspace}' not found in config")
                if display and display not in display_names:
                    warnings.append(f"display '{display}' not found in config")
                if view and display and view not in view_names:
                    warnings.append(f"view '{view}' not valid for display '{display}'")
            except Exception as e:
                warnings.append(f"Could not validate against config: {e}")

        self.parameter_output_values[self._color_params_output.name] = OCIOColorParamsArtifact(
            source_colorspace=source_colorspace,
            display=display,
            view=view,
            config_path=config_file_path,
        )

        if warnings:
            detail = "Emitted with warnings: " + "; ".join(warnings)
        else:
            detail = f"source={source_colorspace!r}, display={display!r}, view={view!r}"

        self._set_status_results(was_successful=True, result_details=detail)

    # --- Private helpers ---

    def _refresh_all_dropdowns(self, file_path: str | None) -> None:
        try:
            config = load_ocio_config(file_path)
            colorspace_names, display_names, role_names = extract_lists(config)
            sc_choices = role_names + colorspace_names

            self._update_param_choices(
                "source_colorspace", sc_choices, self.get_parameter_value("source_colorspace") or ""
            )
            self._update_param_choices("display", display_names, self.get_parameter_value("display") or "")
            # view is populated automatically: _update_option_choices("display", ...) calls
            # set_parameter_value("display", ...) which triggers after_value_set -> _refresh_view_choices

        except Exception as e:
            logger.warning("OCIOColorParameters '%s': could not load config: %s", self.name, e)

    def _refresh_view_choices(self, file_path: str | None, display: str) -> None:
        try:
            config = load_ocio_config(file_path)
            view_names = list(config.getViews(display)) if display else []
            self._update_param_choices("view", view_names, self.get_parameter_value("view") or "")
        except Exception as e:
            logger.warning("OCIOColorParameters '%s': could not refresh views: %s", self.name, e)

    def _clear_dropdowns(self) -> None:
        self._update_option_choices("source_colorspace", [""], "")
        self._update_option_choices("display", [""], "")
        self._update_option_choices("view", [""], "")
        self.hide_message_by_name(
            [
                "invalid_colorspace_message",
                "invalid_display_message",
                "invalid_view_message",
            ]
        )

    def _update_param_choices(self, param_name: str, choices: list[str], current_value: str = "") -> None:
        """Safely call _update_option_choices, falling back to placeholder when choices is empty."""
        safe_choices = choices if choices else [""]
        default = current_value if current_value in safe_choices else safe_choices[0]
        self._update_option_choices(param_name, safe_choices, default)

    def _get_options_choices(self, param: Parameter) -> list[str]:
        """Return current dropdown choices. Used by tests and validation helpers."""
        # "simple_dropdown" is the key written by the Options trait into param.ui_options.
        return list(param.ui_options.get("simple_dropdown", []))

    def _validate_colorspace(self, value: str) -> None:
        choices = self._get_options_choices(self._source_colorspace_param)
        real_choices = [c for c in choices if c]  # exclude placeholder ""
        if value and real_choices and value not in real_choices:
            self._invalid_colorspace_message.value = (
                f"**Unknown colorspace:** `{value}` is not defined in the current OCIO config."
            )
            self.show_message_by_name("invalid_colorspace_message")
        else:
            self.hide_message_by_name("invalid_colorspace_message")

    def _validate_display(self, value: str) -> None:
        choices = self._get_options_choices(self._display_param)
        real_choices = [c for c in choices if c]
        if value and real_choices and value not in real_choices:
            self._invalid_display_message.value = (
                f"**Unknown display:** `{value}` is not defined in the current OCIO config."
            )
            self.show_message_by_name("invalid_display_message")
        else:
            self.hide_message_by_name("invalid_display_message")

    def _validate_view(self, value: str) -> None:
        choices = self._get_options_choices(self._view_param)
        real_choices = [c for c in choices if c]  # exclude placeholder ""
        if value and real_choices and value not in real_choices:
            self._invalid_view_message.value = f"**Unknown view:** `{value}` is not valid for the selected display."
            self.show_message_by_name("invalid_view_message")
        else:
            self.hide_message_by_name("invalid_view_message")
