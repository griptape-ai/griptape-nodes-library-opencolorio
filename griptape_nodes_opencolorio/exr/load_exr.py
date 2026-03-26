"""LoadEXR node for loading OpenEXR files and exposing parts and layers.

Architecture: Two-phase lazy loading via OIIO.
- Phase 1 (after_value_set on file path): Scan headers only (no pixels).
  Populates UI with part previews and layer artifacts.
- Phase 2 (downstream): Nodes use EXRPartArtifact/EXRLayerArtifact
  descriptors to load exactly the channels they need on demand.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterGroup,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.os_events import OpenAssociatedFileRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload, NodeMessageResult
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes_opencolorio.exr.exr import (
    EXRData,
    EXRLayer,
    EXRPart,
    parse_channel_name,
    scan_exr,
)
from griptape_nodes_opencolorio.exr.exr_file_artifact import EXRFileArtifact
from griptape_nodes_opencolorio.exr.exr_layer_artifact import EXRLayerArtifact
from griptape_nodes_opencolorio.exr.exr_part_artifact import EXRPartArtifact

logger = logging.getLogger("griptape_nodes")


# Naming constants for dynamically created elements
_PART_PREFIX = "part_"
_LAYER_PREFIX = "layer_"
_DEFAULT_LAYER_NAME = "default"


def _part_prefix(part: EXRPart, exr_data: EXRData) -> str:
    """Return a disambiguating prefix for multi-part EXR files, empty for single-part."""
    if len(exr_data.parts) > 1:
        return f"p{part.index}_"
    return ""


def _layer_key(layer: EXRLayer, part_prefix: str) -> str:
    """Build the canonical key used to identify a layer in parameter names."""
    return f"{part_prefix}{layer.name or _DEFAULT_LAYER_NAME}"


class LoadEXR(SuccessFailureNode):
    """Load an OpenEXR file and expose its parts and layers as artifacts.

    Outputs EXRFileArtifact, per-part EXRPartArtifact, and per-layer
    EXRLayerArtifact descriptors — downstream nodes use these to load
    only the pixel data they need.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self._cached_exr_data: EXRData | None = None

        # --- Top-level parameters ---

        self._file_path_param = ParameterString(
            name="file_path",
            default_value="",
            tooltip="Path to the EXR file to load",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self._file_path_param.add_trait(
            FileSystemPicker(
                allow_files=True,
                allow_directories=False,
                file_extensions=[".exr"],
            )
        )
        self.add_parameter(self._file_path_param)

        self._open_viewer_param = ParameterButton(
            name="open_in_viewer",
            label="Open in Viewer",
            variant="secondary",
            icon="external-link",
            tooltip="Open the EXR file in your OS default viewer",
            on_click=self._on_open_in_viewer,
        )
        self.add_parameter(self._open_viewer_param)

        # --- Collapsed EXR Info group ---
        with ParameterGroup(name="EXR Info") as exr_info_group:
            exr_info_group.ui_options = {"collapsed": True}

            self._image_width_param = ParameterInt(
                name="image_width",
                display_name="Image Width",
                default_value=0,
                tooltip="Image width in pixels",
                allowed_modes={ParameterMode.OUTPUT},
            )
            self._image_height_param = ParameterInt(
                name="image_height",
                display_name="Image Height",
                default_value=0,
                tooltip="Image height in pixels",
                allowed_modes={ParameterMode.OUTPUT},
            )
            self._compression_param = ParameterString(
                name="compression",
                default_value="",
                tooltip="EXR compression type",
                allowed_modes={ParameterMode.OUTPUT},
            )
            self._part_count_param = ParameterInt(
                name="part_count",
                default_value=0,
                tooltip="Number of parts in the EXR file",
                allowed_modes={ParameterMode.OUTPUT},
            )
            self._layer_count_param = ParameterInt(
                name="layer_count",
                default_value=0,
                tooltip="Total number of layers across all parts",
                allowed_modes={ParameterMode.OUTPUT},
            )
            self._pixel_aspect_ratio_param = ParameterFloat(
                name="pixel_aspect_ratio",
                default_value=1.0,
                tooltip="Pixel aspect ratio",
                allowed_modes={ParameterMode.OUTPUT},
            )
            self._data_window_param = ParameterString(
                name="data_window",
                default_value="",
                tooltip="Data window coordinates (xmin,ymin - xmax,ymax)",
                allowed_modes={ParameterMode.OUTPUT},
            )
            self._display_window_param = ParameterString(
                name="display_window",
                default_value="",
                tooltip="Display window coordinates (xmin,ymin - xmax,ymax)",
                allowed_modes={ParameterMode.OUTPUT},
            )
            self._custom_attributes_param = ParameterString(
                name="custom_attributes",
                default_value="",
                tooltip="Custom EXR header attributes as JSON",
                allowed_modes={ParameterMode.OUTPUT},
            )

        self.add_node_element(exr_info_group)

        # --- Outputs ---

        self._exr_file_param = Parameter(
            name="exr_file",
            display_name="EXR File",
            type="EXRFileArtifact",
            output_type="EXRFileArtifact",
            tooltip="EXR file descriptor (path + headers, no pixel data)",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._exr_file_param)

        self._all_parts_param = Parameter(
            name="all_parts",
            display_name="All Parts",
            type="list[EXRPartArtifact]",
            output_type="list[EXRPartArtifact]",
            tooltip="All parts in the EXR file",
            allowed_modes={ParameterMode.OUTPUT},
            settable=False,
        )
        self.add_parameter(self._all_parts_param)

        self._all_layers_param = Parameter(
            name="all_layers",
            display_name="All Layers",
            type="list[EXRLayerArtifact]",
            output_type="list[EXRLayerArtifact]",
            tooltip="All layers across all parts in the EXR file",
            allowed_modes={ParameterMode.OUTPUT},
            settable=False,
        )
        self.add_parameter(self._all_layers_param)

        # --- Dynamic groups (children populated after scan) ---

        self._parts_group = ParameterGroup(name="exr_parts")
        self._parts_group.ui_options = {"display_name": "Parts"}
        self.add_node_element(self._parts_group)

        self._layers_group = ParameterGroup(name="exr_layers")
        self._layers_group.ui_options = {"display_name": "Layers"}
        self.add_node_element(self._layers_group)

        self._create_status_parameters(
            result_details_tooltip="Details about the EXR load result",
            result_details_placeholder="Load details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    # --- Public instance methods ---

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter is self._file_path_param:
            self._on_file_path_changed(str(value) if value else "")

    async def aprocess(self) -> None:
        """Validate state and set success/failure status."""
        self._clear_execution_status()

        file_path = self.get_parameter_value(self._file_path_param.name)
        if not file_path:
            self._set_status_results(was_successful=False, result_details="No file path provided")
            return

        if not self._cached_exr_data:
            self._set_status_results(was_successful=False, result_details="No EXR data loaded")
            return

        exr_data = self._cached_exr_data
        part = exr_data.parts[0]
        total_layers = sum(len(p.layers) for p in exr_data.parts)
        layer_names: list[str] = []
        for p in exr_data.parts:
            for layer in p.layers:
                layer_names.append(layer.name or _DEFAULT_LAYER_NAME)  # noqa: PERF401
        details = f"Loaded {part.width}x{part.height}, {total_layers} layers: {', '.join(layer_names)}"
        self._set_status_results(was_successful=True, result_details=details)

    # --- Private instance methods (high-level first, helpers below callers) ---

    def _on_open_in_viewer(
        self,
        button: Button,  # noqa: ARG002
        button_details: ButtonDetailsMessagePayload,
    ) -> NodeMessageResult:
        """Open the loaded EXR file in the OS default application."""
        file_path = self.get_parameter_value(self._file_path_param.name)
        if not file_path:
            return NodeMessageResult(
                success=False,
                details="No file path set",
                response=button_details,
                altered_workflow_state=False,
            )

        GriptapeNodes.handle_request(OpenAssociatedFileRequest(path_to_file=file_path))
        return NodeMessageResult(
            success=True,
            details=f"Opening {file_path}",
            response=button_details,
            altered_workflow_state=False,
        )

    def _on_file_path_changed(self, file_path: str) -> None:
        """Scan the EXR header and populate UI. No full pixel load."""
        self._remove_dynamic_elements()
        self._cached_exr_data = None

        if not file_path:
            return

        try:
            exr_data = scan_exr(file_path)
        except (ValueError, RuntimeError) as e:
            logger.error(
                "LoadEXR '%s': Attempted to scan EXR file '%s'. Failed because: %s",
                self.name,
                file_path,
                e,
            )
            return

        self._cached_exr_data = exr_data

        part = exr_data.parts[0]

        self.parameter_output_values[self._image_width_param.name] = part.width
        self.parameter_output_values[self._image_height_param.name] = part.height
        self._populate_exr_info(exr_data)

        file_artifact = self._build_file_artifact(file_path, exr_data)
        self.parameter_output_values[self._exr_file_param.name] = file_artifact

        self._populate_parts_group(file_path, exr_data)
        self._populate_layers_group(file_path, exr_data)

    def _populate_exr_info(self, exr_data: EXRData) -> None:
        """Populate the EXR Info output parameters from the first part's header."""
        header = exr_data.parts[0].header

        self.parameter_output_values[self._compression_param.name] = header.compression.value
        self.parameter_output_values[self._part_count_param.name] = len(exr_data.parts)
        self.parameter_output_values[self._layer_count_param.name] = sum(len(p.layers) for p in exr_data.parts)
        self.parameter_output_values[self._pixel_aspect_ratio_param.name] = header.pixel_aspect_ratio

        dw = header.data_window
        self.parameter_output_values[self._data_window_param.name] = f"{dw.xmin},{dw.ymin} - {dw.xmax},{dw.ymax}"

        disp = header.display_window
        self.parameter_output_values[self._display_window_param.name] = (
            f"{disp.xmin},{disp.ymin} - {disp.xmax},{disp.ymax}"
        )

        self.parameter_output_values[self._custom_attributes_param.name] = (
            json.dumps(header.custom, indent=2, default=str) if header.custom else "{}"
        )

    def _populate_parts_group(self, file_path: str, exr_data: EXRData) -> None:
        """Populate the Parts group with per-part artifact outputs, update All Parts."""
        all_part_artifacts: list[EXRPartArtifact] = []
        is_multi_part = len(exr_data.parts) > 1

        for part in exr_data.parts:
            part_artifact = self._build_part_artifact(file_path, part.index)
            all_part_artifacts.append(part_artifact)

            part_display = self._build_part_display_name(part, is_multi_part=is_multi_part)

            part_param = Parameter(
                name=f"{_PART_PREFIX}{part.index}",
                display_name=part_display,
                type="EXRPartArtifact",
                output_type="EXRPartArtifact",
                tooltip=f"Descriptor for part {part.index} ({part.width}x{part.height})",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False,
            )
            self._parts_group.add_child(part_param)
            self.parameter_output_values[part_param.name] = part_artifact

        self.parameter_output_values[self._all_parts_param.name] = all_part_artifacts

    def _populate_layers_group(self, file_path: str, exr_data: EXRData) -> None:
        """Populate the Layers group with per-layer artifacts, update All Layers."""
        all_layer_artifacts: list[EXRLayerArtifact] = []

        for part in exr_data.parts:
            prefix = _part_prefix(part, exr_data)
            for layer in part.layers:
                key = _layer_key(layer, prefix)
                layer_artifact = self._build_layer_artifact(file_path, part.index, layer)
                all_layer_artifacts.append(layer_artifact)

                layer_display = layer.name or _DEFAULT_LAYER_NAME
                channel_names = [parse_channel_name(ch.name).channel_name for ch in layer.channels]
                channels_str = ", ".join(channel_names)

                layer_param = Parameter(
                    name=f"{_LAYER_PREFIX}{key}",
                    display_name=f"{layer_display} ({channels_str})",
                    type="str",
                    output_type="EXRLayerArtifact",
                    default_value=layer_display,
                    tooltip=f"Layer '{layer_display}' with channels: {channels_str}",
                    allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                    settable=False,
                    hide_property=True,
                )
                self._layers_group.add_child(layer_param)
                self.parameter_output_values[layer_param.name] = layer_artifact

        self.parameter_output_values[self._all_layers_param.name] = all_layer_artifacts

    def _build_part_display_name(self, part: EXRPart, *, is_multi_part: bool) -> str:
        """Build a human-readable display name for a part parameter.

        Single-part: "Single Part" or "Single Part: rgba"
        Multi-part:  "Part 1" or "Part 1: rgba"
        """
        part_name = part.header.name

        if is_multi_part:
            label = f"Part {part.index + 1}"
        else:
            label = "Single Part"

        if part_name:
            return f"{label}: {part_name}"
        return label

    def _build_file_artifact(self, file_path: str, exr_data: EXRData) -> EXRFileArtifact:
        """Build an EXRFileArtifact from scanned EXR data."""
        parts: list[EXRPartArtifact] = []
        for part in exr_data.parts:
            part_artifact = self._build_part_artifact(file_path, part.index)
            parts.append(part_artifact)
        return EXRFileArtifact(file_path=file_path, parts=parts)

    def _build_layer_artifact(self, file_path: str, part_index: int, layer: EXRLayer) -> EXRLayerArtifact:
        """Build an EXRLayerArtifact composing a part artifact with a layer info."""
        part_artifact = self._build_part_artifact(file_path, part_index)
        return EXRLayerArtifact(part=part_artifact, layer=layer)

    def _build_part_artifact(self, file_path: str, part_index: int) -> EXRPartArtifact:
        """Build an EXRPartArtifact from cached EXR data."""
        exr_data = self._cached_exr_data
        if not exr_data:
            msg = f"LoadEXR '{self.name}': Attempted to build part artifact. Failed because no cached EXR data is available."
            raise RuntimeError(msg)

        part = exr_data.parts[part_index]

        return EXRPartArtifact(
            file_path=file_path,
            part_index=part.index,
            name=part.header.name,
            width=part.width,
            height=part.height,
            header=part.header,
            channels=part.channels,
            layers=part.layers,
        )

    def _remove_dynamic_elements(self) -> None:
        """Remove all dynamic children from Parts and Layers groups."""
        for child in list(self._parts_group.children):
            self._parts_group.remove_child(child)
        for child in list(self._layers_group.children):
            self._layers_group.remove_child(child)

