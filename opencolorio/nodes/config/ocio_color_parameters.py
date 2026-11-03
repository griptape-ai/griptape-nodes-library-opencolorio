from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

from opencolorio.artifacts.ocio_color_params_artifact import OCIOColorParamsArtifact
from opencolorio.artifacts.ocio_config_artifact import OCIOConfigArtifact
from opencolorio.defaults import DEFAULT_SENTINEL, resolve_color_params
from opencolorio.ocio_helpers import extract_lists, load_ocio_config

# Which layer answered, phrased for the badge popover rather than for a log.
_SOURCE_DESCRIPTIONS = {
    "explicit": "the value you chose",
    "settings": "your OpenColorIO library settings",
    "settings-invalid": "the config, because the library setting names something it does not define",
    "config-role": "the config's scene_linear role",
    "config-file-rule": "the config's File Rules for this path",
    "config-default-rule": "the config's Default file rule",
    "config-default-display": "the config's default display",
    "config-default-view": "the config's default view for this display",
    "fallback": "nothing: the config had no answer",
    "no-config": "no OCIO config, so colour management is off",
}

logger = logging.getLogger("griptape_nodes")


class OCIOColorParameters(SuccessFailureNode):
    """Bundle source colorspace, display, and view into a reusable OCIOColorParamsArtifact.

    When an OCIOConfigArtifact is wired to the ``config`` input the three
    dropdowns are populated from that config and restricted to valid combinations.
    Without a config, dropdowns show a placeholder and emit empty-string values.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # _update_option_choices calls set_parameter_value, which fires after_value_set,
        # which would ask for hints again partway through a refresh.
        self._refreshing_hints = False

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
            default_value=DEFAULT_SENTINEL,
            tooltip="Source colorspace name selected from the connected OCIO config",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[DEFAULT_SENTINEL], allow_custom=True)},
        )
        self.add_parameter(self._source_colorspace_param)

        self._display_param = ParameterString(
            name="display",
            display_name="Display",
            default_value=DEFAULT_SENTINEL,
            tooltip="Display device name selected from the connected OCIO config",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[DEFAULT_SENTINEL], allow_custom=True)},
        )
        self.add_parameter(self._display_param)

        self._view_param = ParameterString(
            name="view",
            display_name="View",
            default_value=DEFAULT_SENTINEL,
            tooltip="View name for the selected display",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[DEFAULT_SENTINEL], allow_custom=True)},
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

        self._refresh_default_hints()

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

        if parameter in (self._config_param, self._source_colorspace_param, self._display_param, self._view_param):
            self._refresh_default_hints()

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
                self._refresh_default_hints()
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
            self._refresh_default_hints()
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def process(self) -> None:
        self._clear_execution_status()

        source_colorspace = self.get_parameter_value("source_colorspace") or ""
        display = self.get_parameter_value("display") or ""
        view = self.get_parameter_value("view") or ""

        self._reject_blank_parameters({"Source Colorspace": source_colorspace, "Display": display, "View": view})

        # Read config state from metadata — set by after_value_set when the framework
        # propagates the upstream value. parameter_output_values is cleared before
        # execution so cannot be used here.
        config_connected: bool = self.metadata.get("_config_connected", False)
        config_file_path: str | None = self.metadata.get("_config_file_path") if config_connected else None
        warnings: list[str] = []
        provenance: list[str] = []

        if DEFAULT_SENTINEL in (source_colorspace, display, view):
            source_colorspace, display, view = self._resolve_sentinels(
                source_colorspace=source_colorspace,
                display=display,
                view=view,
                config_file_path=config_file_path,
                warnings=warnings,
                provenance=provenance,
            )

        if config_connected:
            warnings.extend(self._validate_against_config(config_file_path, source_colorspace, display, view))

        self.parameter_output_values[self._color_params_output.name] = OCIOColorParamsArtifact(
            source_colorspace=source_colorspace,
            display=display,
            view=view,
            config_path=config_file_path,
        )

        detail = f"source={source_colorspace!r}, display={display!r}, view={view!r}"
        if provenance:
            detail += " (" + "; ".join(provenance) + ")"
        if warnings:
            detail += ". Warnings: " + "; ".join(warnings)

        self._set_status_results(was_successful=True, result_details=detail)

    def _reject_blank_parameters(self, values: dict[str, str]) -> None:
        """Fail on a blank parameter, naming the fix.

        Blank is not migrated to the sentinel. Doing that silently would change
        what a saved workflow renders, where failing says plainly what to pick.
        """
        blank = [label for label, value in values.items() if not value]
        if not blank:
            return

        names = ", ".join(blank)
        message = (
            f"{names} is empty. Set it to {DEFAULT_SENTINEL} to use the OCIO config's own answer, "
            "or choose a value from the dropdown."
        )
        self._set_status_results(was_successful=False, result_details=message)
        self._handle_failure_exception(ValueError(message))

    def _resolve_sentinels(
        self,
        *,
        source_colorspace: str,
        display: str,
        view: str,
        config_file_path: str | None,
        warnings: list[str],
        provenance: list[str],
    ) -> tuple[str, str, str]:
        """Replace each sentinel with the resolver's answer, recording where it came from."""
        explicit = OCIOColorParamsArtifact(
            source_colorspace="" if source_colorspace == DEFAULT_SENTINEL else source_colorspace,
            display="" if display == DEFAULT_SENTINEL else display,
            view="" if view == DEFAULT_SENTINEL else view,
            config_path=config_file_path,
        )

        resolved = resolve_color_params(explicit=explicit, config_path=config_file_path)

        warnings.extend(resolved.warnings)
        for slot, current in (
            ("source_colorspace", source_colorspace),
            ("display", display),
            ("view", view),
        ):
            if current == DEFAULT_SENTINEL:
                provenance.append(f"{slot} from {resolved.sources.get(slot, 'unknown')}")

        return (
            resolved.source_colorspace if source_colorspace == DEFAULT_SENTINEL else source_colorspace,
            resolved.display if display == DEFAULT_SENTINEL else display,
            resolved.view if view == DEFAULT_SENTINEL else view,
        )

    def _validate_against_config(
        self,
        config_file_path: str | None,
        source_colorspace: str,
        display: str,
        view: str,
    ) -> list[str]:
        """Re-check the emitted values against the live config, without blocking the emit."""
        warnings: list[str] = []
        try:
            config = load_ocio_config(config_file_path)
            colorspace_names = list(config.getColorSpaceNames())
            role_names = [r for r, _ in config.getRoles()]
            display_names = list(config.getDisplays())
            view_names = list(config.getViews(display)) if display else []

            if source_colorspace and source_colorspace not in colorspace_names and source_colorspace not in role_names:
                warnings.append(f"source_colorspace '{source_colorspace}' not found in config")
            if display and display not in display_names:
                warnings.append(f"display '{display}' not found in config")
            if view and display and view not in view_names:
                warnings.append(f"view '{view}' not valid for display '{display}'")
        except Exception as e:  # noqa: BLE001 - a config that will not load is a warning, not a failed emit
            warnings.append(f"Could not validate against config: {e}")
        return warnings

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
        self._update_option_choices("source_colorspace", [DEFAULT_SENTINEL], DEFAULT_SENTINEL)
        self._update_option_choices("display", [DEFAULT_SENTINEL], DEFAULT_SENTINEL)
        self._update_option_choices("view", [DEFAULT_SENTINEL], DEFAULT_SENTINEL)
        self.hide_message_by_name(
            [
                "invalid_colorspace_message",
                "invalid_display_message",
                "invalid_view_message",
            ]
        )

    def _update_param_choices(self, param_name: str, choices: list[str], current_value: str = "") -> None:
        """Call _update_option_choices with the sentinel always offered first.

        The sentinel replaces the old empty-string placeholder, so the dropdown
        still renders when the config has nothing to offer, and "use the default"
        is a real choice rather than something blank implies.
        """
        safe_choices = [DEFAULT_SENTINEL, *choices]
        default = current_value if current_value in safe_choices else DEFAULT_SENTINEL
        self._update_option_choices(param_name, safe_choices, default)

    def _get_options_choices(self, param: Parameter) -> list[str]:
        """Return current dropdown choices. Used by tests and validation helpers."""
        # "simple_dropdown" is the key written by the Options trait into param.ui_options.
        return list(param.ui_options.get("simple_dropdown", []))

    def _validate_colorspace(self, value: str) -> None:
        if value == DEFAULT_SENTINEL:
            self.hide_message_by_name("invalid_colorspace_message")
            return
        choices = self._get_options_choices(self._source_colorspace_param)
        real_choices = [c for c in choices if c and c != DEFAULT_SENTINEL]
        if value and real_choices and value not in real_choices:
            self._invalid_colorspace_message.value = (
                f"**Unknown colorspace:** `{value}` is not defined in the current OCIO config."
            )
            self.show_message_by_name("invalid_colorspace_message")
        else:
            self.hide_message_by_name("invalid_colorspace_message")

    def _validate_display(self, value: str) -> None:
        if value == DEFAULT_SENTINEL:
            self.hide_message_by_name("invalid_display_message")
            return
        choices = self._get_options_choices(self._display_param)
        real_choices = [c for c in choices if c and c != DEFAULT_SENTINEL]
        if value and real_choices and value not in real_choices:
            self._invalid_display_message.value = (
                f"**Unknown display:** `{value}` is not defined in the current OCIO config."
            )
            self.show_message_by_name("invalid_display_message")
        else:
            self.hide_message_by_name("invalid_display_message")

    def _validate_view(self, value: str) -> None:
        if value == DEFAULT_SENTINEL:
            self.hide_message_by_name("invalid_view_message")
            return
        choices = self._get_options_choices(self._view_param)
        real_choices = [c for c in choices if c and c != DEFAULT_SENTINEL]
        if value and real_choices and value not in real_choices:
            self._invalid_view_message.value = f"**Unknown view:** `{value}` is not valid for the selected display."
            self.show_message_by_name("invalid_view_message")
        else:
            self.hide_message_by_name("invalid_view_message")

    # --- Default hints ---

    def _refresh_default_hints(self) -> None:
        """Show what each <default> parameter will resolve to, and why.

        The placeholder answers "what will this be?" and the badge answers
        "why?", both at zero row height. Neither writes into the parameter
        value, so the sentinel still goes over the wire and a saved workflow
        keeps resolving at run time rather than freezing today's answer.

        Resolution reads a config from disk, so this is best-effort: a hint must
        never stop a node being constructed.
        """
        if self._refreshing_hints:
            return

        params = {
            "source_colorspace": self._source_colorspace_param,
            "display": self._display_param,
            "view": self._view_param,
        }
        sentinel_slots = [
            slot for slot, param in params.items() if self.get_parameter_value(param.name) == DEFAULT_SENTINEL
        ]

        for slot, param in params.items():
            if slot not in sentinel_slots:
                self._clear_hint(param)

        if not sentinel_slots:
            return

        config_file_path: str | None = (
            self.metadata.get("_config_file_path") if self.metadata.get("_config_connected") else None
        )

        self._refreshing_hints = True
        try:
            resolved = resolve_color_params(config_path=config_file_path)
        except Exception as e:  # noqa: BLE001 - a hint is never worth failing over
            logger.debug("OCIOColorParameters '%s': could not resolve default hints: %s", self.name, e)
            for slot in sentinel_slots:
                self._clear_hint(params[slot])
            return
        finally:
            self._refreshing_hints = False

        config_label = resolved.config_path or ("$OCIO" if config_file_path is None else config_file_path)
        for slot in sentinel_slots:
            value = getattr(resolved, slot, "")
            source = resolved.sources.get(slot, "")
            self._set_hint(params[slot], value=value, source=source, config_label=config_label)

    def _set_hint(self, param: Parameter, *, value: str, source: str, config_label: str | None) -> None:
        explanation = _SOURCE_DESCRIPTIONS.get(source, source or "an unknown source")
        shown = value or "nothing"
        where = f" in `{config_label}`" if config_label else ""

        param.ui_options = {
            **param.ui_options,
            "placeholder_text": value,
            "badge": {
                "variant": "info",
                "title": "Default",
                "message": f"Resolves to `{shown}`, from {explanation}{where}.",
                "hide": False,
                # Without this the dismiss button round-trips to the backend.
                "hide_clear_button": True,
            },
        }

    @staticmethod
    def _clear_hint(param: Parameter) -> None:
        options = {k: v for k, v in param.ui_options.items() if k not in ("placeholder_text", "badge")}
        param.ui_options = options
