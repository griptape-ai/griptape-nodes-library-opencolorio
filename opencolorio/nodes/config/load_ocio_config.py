from __future__ import annotations

import logging
import os
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.file_system_picker import FileSystemPicker

from opencolorio.artifacts.ocio_config_artifact import OCIOConfigArtifact
from opencolorio.ocio_helpers import extract_lists, load_ocio_config

logger = logging.getLogger("griptape_nodes")


class LoadOCIOConfig(SuccessFailureNode):
    """Load an OpenColorIO config from the $OCIO environment variable or an explicit override path.

    The $OCIO environment variable is the primary source of truth and is re-read live at
    each execution, so Griptape project-level OCIO overrides (project.yml `environment:`)
    are always honoured. An "Advanced" group provides an explicit path override for
    development and testing workflows.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        current_ocio = os.environ.get("OCIO", "")
        saved_override_enabled: bool = self.metadata.get("_override_enabled", False)

        # --- Detection messages ---
        self._ocio_detected_message = ParameterMessage(
            name="ocio_detected_message",
            variant="info",
            value=self._ocio_detected_text(current_ocio),
            markdown=True,
            hide=not bool(current_ocio),
        )
        self.add_node_element(self._ocio_detected_message)

        self._ocio_missing_message = ParameterMessage(
            name="ocio_missing_message",
            variant="warning",
            value=(
                "No `$OCIO` environment variable detected.\n\n"
                "Set `$OCIO` in your environment or project settings, "
                "or use **Advanced → Override OCIO Config** to specify a path directly."
            ),
            markdown=True,
            hide=bool(current_ocio),
        )
        self.add_node_element(self._ocio_missing_message)

        # Shown on workflow reload when $OCIO differs from the value at last successful run.
        saved_ocio_env = self.metadata.get("_saved_ocio_env", "")
        ocio_changed = not saved_override_enabled and bool(saved_ocio_env) and current_ocio != saved_ocio_env
        self._ocio_changed_message = ParameterMessage(
            name="ocio_changed_message",
            variant="warning",
            value=(
                "**$OCIO changed since last run.**\n\n"
                f"Was: `{saved_ocio_env}`\n\n"
                f"Now: `{current_ocio}`\n\n"
                "Re-run this node to update downstream outputs."
            ),
            markdown=True,
            hide=not ocio_changed,
        )
        self.add_node_element(self._ocio_changed_message)

        # Shown when the override toggle is ON — replaces the env var messages.
        self._ocio_override_active_message = ParameterMessage(
            name="ocio_override_active_message",
            variant="info",
            value=(
                "**OCIO Override active** — using explicit config file.\n\n"
                "Disable **Override OCIO Config** to use the `$OCIO` global config instead."
            ),
            markdown=True,
            hide=not saved_override_enabled,
        )
        self.add_node_element(self._ocio_override_active_message)

        # When restoring with override ON, hide the env var messages so only the override
        # message is shown.
        if saved_override_enabled:
            self.hide_message_by_name(["ocio_detected_message", "ocio_missing_message", "ocio_changed_message"])

        # --- Context variables (always visible) ---
        self._context_vars_param = Parameter(
            name="context_vars",
            display_name="Context Variables",
            type="dict",
            default_value={},
            tooltip='OCIO context variables, e.g. {"SHOT": "sh010", "SEQ": "sq020"}',
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(self._context_vars_param)

        # --- Advanced group ---
        advanced_group = ParameterGroup(name="Advanced", ui_options={"collapsed": True})
        with advanced_group:
            self._override_param = ParameterBool(
                name="override_ocio_config",
                display_name="Override OCIO Config",
                default_value=saved_override_enabled,
                tooltip=(
                    "When enabled, use the explicit file path below instead of the $OCIO "
                    "environment variable. Useful during development when you need to test "
                    "a specific config without changing your environment."
                ),
                allowed_modes={ParameterMode.PROPERTY},
            )

            self._file_path_param = ParameterString(
                name="file_path",
                display_name="Config File Path",
                default_value="",
                tooltip="Path to an OCIO config file (.ocio). Only used when Override OCIO Config is enabled.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                hide=not saved_override_enabled,
            )
            self._file_path_param.add_trait(
                FileSystemPicker(
                    allow_files=True,
                    allow_directories=False,
                    file_extensions=[".ocio"],
                    initial_path=current_ocio,
                )
            )
        self.add_node_element(advanced_group)

        # --- Output ---
        self._config_param = Parameter(
            name="config",
            display_name="OCIO Config",
            type="OCIOConfigArtifact",
            output_type="OCIOConfigArtifact",
            tooltip="OCIOConfigArtifact carrying the resolved config path and context variables",
            allowed_modes={ParameterMode.OUTPUT},
            settable=False,
        )
        self.add_parameter(self._config_param)

        self._create_status_parameters(
            result_details_tooltip="Details about the OCIO config load result",
            result_details_placeholder="Config details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    # --- Lifecycle ---

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter is self._override_param:
            enabled = bool(value)
            if enabled:
                self.show_parameter_by_name("file_path")
                self.hide_message_by_name(["ocio_detected_message", "ocio_missing_message", "ocio_changed_message"])
                self.show_message_by_name("ocio_override_active_message")
            else:
                self.hide_parameter_by_name("file_path")
                self.hide_message_by_name("ocio_override_active_message")
                # Restore the appropriate env var message, refreshing content in case
                # $OCIO changed while override was active.
                live_ocio = os.environ.get("OCIO", "")
                if live_ocio:
                    self._ocio_detected_message.value = self._ocio_detected_text(live_ocio)
                    self.show_message_by_name("ocio_detected_message")
                else:
                    self.show_message_by_name("ocio_missing_message")
            self.metadata["_override_enabled"] = enabled
            self._set_status_results(was_successful=False, result_details="Re-run to update outputs.")

        elif parameter is self._file_path_param:
            raw = str(value) if value else None
            try:
                self._resolve_path(raw)
            except FileLoadError as e:
                logger.warning("LoadOCIOConfig '%s': could not resolve path '%s': %s", self.name, raw, e)
                self.parameter_output_values[self._config_param.name] = None
                return super().after_value_set(parameter, value)

            override_enabled = self.get_parameter_value(self._override_param.name) or False
            if override_enabled:
                self._set_status_results(was_successful=False, result_details="Re-run to update outputs.")

        return super().after_value_set(parameter, value)

    def process(self) -> None:
        self._clear_execution_status()

        try:
            override_enabled = bool(self.get_parameter_value(self._override_param.name))
            raw_path = self.get_parameter_value(self._file_path_param.name) or None

            # Re-read $OCIO live — honours Griptape project-level env overrides.
            live_ocio = os.environ.get("OCIO", "")
            effective = self._effective_path(override_enabled, raw_path)

            context_vars = self.get_parameter_value(self._context_vars_param.name) or {}

            config = load_ocio_config(effective)
            colorspace_names, display_names, role_names = extract_lists(config)

            self.parameter_output_values[self._config_param.name] = OCIOConfigArtifact(
                file_path=effective,
                context_vars=dict(context_vars),
            )

            # Record the env var used so reload can detect staleness.
            # Clear it when running in override mode so stale values don't trigger false warnings.
            if not override_enabled:
                self.metadata["_saved_ocio_env"] = live_ocio
            else:
                self.metadata.pop("_saved_ocio_env", None)

            # Clear any staleness warning now that the node has run successfully.
            self.hide_message_by_name("ocio_changed_message")

            src = effective or "$OCIO"
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

    @staticmethod
    def _ocio_detected_text(ocio: str) -> str:
        return (
            f"Active $OCIO: `{ocio}`\n\n"
            "This is sampled at node load. The value is re-read live at each execution, "
            "so Griptape project-level OCIO overrides always take effect."
        )

    def _effective_path(self, override_enabled: bool, raw_path: str | None) -> str | None:
        """Return the file path to load from, or None to use $OCIO env var.

        Raises FileLoadError if override is enabled and the path cannot be resolved,
        so callers surface the error rather than silently falling back to $OCIO.
        """
        if override_enabled and raw_path:
            return self._resolve_path(raw_path)
        return None

    def _resolve_path(self, raw: str | None) -> str | None:
        if not raw:
            return None
        return File(raw).resolve()
