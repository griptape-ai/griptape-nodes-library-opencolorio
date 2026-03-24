"""LoadEXR node for loading OpenEXR files and exposing layers."""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from io import BytesIO
from typing import Any

import httpx
from griptape.artifacts import ImageUrlArtifact

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterGroup,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.os_events import OpenAssociatedFileRequest
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlRequest,
    CreateStaticFileDownloadUrlResultFailure,
    CreateStaticFileDownloadUrlResultSuccess,
    CreateStaticFileUploadUrlRequest,
    CreateStaticFileUploadUrlResultFailure,
    CreateStaticFileUploadUrlResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload, NodeMessageResult
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options
from PIL import Image

from griptape_nodes_opencolorio.exr.exr import (
    EXRData,
    EXRLayer,
    attempt_read_exr,
    generate_exr_preview,
    generate_layer_preview,
    parse_channel_name,
)

logger = logging.getLogger("griptape_nodes")


class ToneMapping(StrEnum):
    SIMPLE = "simple"
    REINHARD = "reinhard"
    FILMIC = "filmic"


# Naming constants for dynamically created elements
_LAYER_PREFIX = "layer_"
_PART_PREFIX = "part_"
_LAYER_PREVIEW_SUFFIX = "_preview"
_LAYER_CHANNELS_SUFFIX = "_channels"
_LAYER_DATA_SUFFIX = "_data"
_PART_PREVIEW_SUFFIX = "_composite"
_DEFAULT_LAYER_NAME = "default"


def _part_prefix(part: Any, exr_data: EXRData) -> str:
    """Return a disambiguating prefix for multi-part EXR files, empty for single-part."""
    if len(exr_data.parts) > 1:
        return f"p{part.index}_"
    return ""


def _layer_key(layer: EXRLayer, part_prefix: str) -> str:
    """Build the canonical key used to identify a layer in the selector and group names."""
    return f"{part_prefix}{layer.name or _DEFAULT_LAYER_NAME}"


class LoadEXR(SuccessFailureNode):
    """Load an OpenEXR file and expose its layers, channels, and metadata."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self._cached_exr_data: EXRData | None = None
        self._cached_tone_mapping = ToneMapping.SIMPLE
        self._layer_groups: dict[str, ParameterGroup] = {}

        # --- Top-level parameters ---

        # File path input with file picker
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

        # Tone mapping selector
        self._tone_mapping_param = ParameterString(
            name="tone_mapping",
            default_value=ToneMapping.SIMPLE.value,
            tooltip="Tone mapping method for sRGB preview generation",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[m.value for m in ToneMapping])},
        )
        self.add_parameter(self._tone_mapping_param)

        # Open in external viewer button
        self._open_viewer_param = Parameter(
            name="open_in_viewer",
            type="str",
            default_value="",
            tooltip="Open the EXR file in your OS default viewer",
            allowed_modes={ParameterMode.PROPERTY},
            traits={
                Button(
                    label="Open in Viewer",
                    variant="secondary",
                    icon="external-link",
                    on_click=self._on_open_in_viewer,
                )
            },
        )
        self.add_parameter(self._open_viewer_param)

        # EXR Part Preview group (children populated dynamically after load)
        self._part_preview_group = ParameterGroup(name="exr_part_preview")
        self._part_preview_group.ui_options = {"display_name": "EXR Part Previews"}
        self.add_node_element(self._part_preview_group)

        # Layer selector (populated after EXR load)
        self._view_layer_param = ParameterString(
            name="view_layer",
            display_name="Select Layer",
            default_value="",
            tooltip="Choose which EXR layer to display",
            allowed_modes={ParameterMode.PROPERTY},
            traits={Options(choices=[])},
        )
        self.add_parameter(self._view_layer_param)

        # Full EXR data output for downstream nodes
        self._exr_data_param = Parameter(
            name="exr_data",
            display_name="EXR Content",
            type="EXRData",
            output_type="EXRData",
            tooltip="Full HDR EXR data structure for downstream processing",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._exr_data_param)

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

        # Status parameters
        self._create_status_parameters(
            result_details_tooltip="Details about the EXR load result",
            result_details_placeholder="Load details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    # --- Button callback ---

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

        request = OpenAssociatedFileRequest(path_to_file=file_path)
        GriptapeNodes.handle_request(request)

        return NodeMessageResult(
            success=True,
            details=f"Opening {file_path}",
            response=button_details,
            altered_workflow_state=False,
        )

    # --- Process ---

    async def aprocess(self) -> None:
        """Load the EXR file and populate all outputs."""
        self._clear_execution_status()

        # FAILURE: no file path
        file_path = self.get_parameter_value(self._file_path_param.name)
        if not file_path:
            self._set_status_results(was_successful=False, result_details="No file path provided")
            return

        # FAILURE: load EXR
        try:
            exr_data = attempt_read_exr(str(file_path))
        except Exception as e:
            error_msg = f"Failed to load EXR: {e}"
            self._set_status_results(was_successful=False, result_details=error_msg)
            logger.error("LoadEXR '%s': %s", self.name, error_msg)
            return

        tone_mapping = self.get_parameter_value(self._tone_mapping_param.name) or ToneMapping.SIMPLE

        # Clean up any previous dynamic elements
        self._remove_dynamic_elements()

        # Cache for layer visibility toggling
        self._cached_exr_data = exr_data
        self._cached_tone_mapping = ToneMapping(tone_mapping)

        part = exr_data.parts[0]

        # Set static outputs
        self.parameter_output_values[self._image_width_param.name] = part.width
        self.parameter_output_values[self._image_height_param.name] = part.height
        self.parameter_output_values[self._exr_data_param.name] = exr_data

        self._populate_exr_info(exr_data)
        await self._populate_part_previews(exr_data, tone_mapping)
        await self._populate_layers(exr_data, tone_mapping)

        # SUCCESS
        total_layers = sum(len(p.layers) for p in exr_data.parts)
        layer_names = [
            layer.name or _DEFAULT_LAYER_NAME
            for p in exr_data.parts
            for layer in p.layers
        ]
        details = f"Loaded {part.width}x{part.height}, {total_layers} layers: {', '.join(layer_names)}"
        self._set_status_results(was_successful=True, result_details=details)

    # --- Value change hook ---

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter is self._view_layer_param:
            self._show_selected_layer(str(value) if value else "")

    # --- EXR info ---

    def _populate_exr_info(self, exr_data: EXRData) -> None:
        """Populate the EXR Info output parameters from the first part's header."""
        header = exr_data.parts[0].header

        self.parameter_output_values[self._compression_param.name] = header.compression.value
        self.parameter_output_values[self._part_count_param.name] = len(exr_data.parts)
        self.parameter_output_values[self._layer_count_param.name] = sum(
            len(p.layers) for p in exr_data.parts
        )
        self.parameter_output_values[self._pixel_aspect_ratio_param.name] = header.pixel_aspect_ratio

        dw = header.data_window
        self.parameter_output_values[self._data_window_param.name] = (
            f"{dw.xmin},{dw.ymin} - {dw.xmax},{dw.ymax}"
        )

        disp = header.display_window
        self.parameter_output_values[self._display_window_param.name] = (
            f"{disp.xmin},{disp.ymin} - {disp.xmax},{disp.ymax}"
        )

        self.parameter_output_values[self._custom_attributes_param.name] = json.dumps(
            header.custom, indent=2, default=str
        ) if header.custom else "{}"

    # --- Part composite previews ---

    def _clear_part_previews(self) -> None:
        """Remove all dynamically created preview children from the part preview group."""
        children = list(self._part_preview_group.find_elements_by_type(ParameterImage))
        for child in children:
            self._part_preview_group.remove_child(child)

    async def _populate_part_previews(self, exr_data: EXRData, tone_mapping: str) -> None:
        """Generate a composite preview image for each part inside the static group."""
        self._clear_part_previews()

        is_multi_part = len(exr_data.parts) > 1
        if is_multi_part:
            self._part_preview_group.ui_options = {"display_name": "EXR Part Previews"}
        else:
            self._part_preview_group.ui_options = {"display_name": "EXR Part Preview"}

        for part in exr_data.parts:
            if is_multi_part:
                preview_display = f"Part {part.index}"
            else:
                preview_display = "Preview"

            preview_param = ParameterImage(
                name=f"{_PART_PREFIX}{part.index}{_PART_PREVIEW_SUFFIX}",
                display_name=preview_display,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip=f"Tone-mapped sRGB composite preview of part {part.index}",
                settable=False,
            )
            self._part_preview_group.add_child(preview_param)

            try:
                preview_image = generate_exr_preview(
                    exr_data, part_index=part.index, tone_mapping_method=tone_mapping  # type: ignore[arg-type]
                )
                preview_artifact = await self._upload_pil_image(
                    preview_image, f"{self.name}_{_PART_PREFIX}{part.index}_preview.png"
                )
                self.parameter_output_values[preview_param.name] = preview_artifact
            except Exception as e:
                logger.warning(
                    "LoadEXR '%s': Composite preview for part %d failed: %s",
                    self.name,
                    part.index,
                    e,
                )
                self.parameter_output_values[preview_param.name] = None

    # --- Layer management ---

    async def _populate_layers(self, exr_data: EXRData, tone_mapping: str) -> None:
        """Create all layer groups (hidden) and set up the layer selector."""
        self._layer_groups = {}
        layer_choices: list[str] = []

        for part in exr_data.parts:
            prefix = _part_prefix(part, exr_data)
            for layer in part.layers:
                key = _layer_key(layer, prefix)
                layer_choices.append(key)
                group = await self._create_layer_group(layer, prefix, tone_mapping)
                self._layer_groups[key] = group

        # Update the Options trait with the new choices
        existing_traits = self._view_layer_param.find_elements_by_type(Options)
        for trait in existing_traits:
            self._view_layer_param.remove_trait(trait_type=trait)
        self._view_layer_param.add_trait(Options(choices=layer_choices))

        # Default to first layer
        default_layer = layer_choices[0] if layer_choices else ""
        self.set_parameter_value(self._view_layer_param.name, default_layer)
        self._show_selected_layer(default_layer)

    async def _create_layer_group(
        self,
        layer: EXRLayer,
        part_prefix: str,
        tone_mapping: str,
    ) -> ParameterGroup:
        """Create a ParameterGroup for a single EXR layer with preview and data outputs."""
        layer_display_name = layer.name or _DEFAULT_LAYER_NAME
        group_name = f"{_LAYER_PREFIX}{_layer_key(layer, part_prefix)}"

        channel_names = [parse_channel_name(ch.name).channel_name for ch in layer.channels]

        channels_str = ", ".join(channel_names)

        layer_group = ParameterGroup(name=group_name, collapsed=True)
        layer_group.ui_options = {
            "collapsed": True,
            "hide": True,
            "display_name": f"{part_prefix}{layer_display_name} (Channels: {channels_str})",
        }

        # Layer preview image
        layer_preview_param = ParameterImage(
            name=f"{group_name}{_LAYER_PREVIEW_SUFFIX}",
            display_name="Layer Preview",
            allowed_modes={ParameterMode.OUTPUT},
            tooltip=f"sRGB preview of layer '{layer_display_name}'",
            settable=False,
        )
        layer_group.add_child(layer_preview_param)

        # Channel names (informational)
        channels_param = Parameter(
            name=f"{group_name}{_LAYER_CHANNELS_SUFFIX}",
            display_name="Channels",
            type="list",
            default_value=channel_names,
            tooltip=f"Channel names in layer '{layer_display_name}'",
            allowed_modes={ParameterMode.OUTPUT},
        )
        layer_group.add_child(channels_param)

        # Layer data output (HDR)
        layer_data_param = Parameter(
            name=f"{group_name}{_LAYER_DATA_SUFFIX}",
            display_name="Pixel Data",
            type="EXRLayer",
            output_type="EXRLayer",
            tooltip=f"HDR float32 data for layer '{layer_display_name}'",
            allowed_modes={ParameterMode.OUTPUT},
        )
        layer_group.add_child(layer_data_param)

        self.add_node_element(layer_group)

        # Set output values
        self.parameter_output_values[channels_param.name] = channel_names
        self.parameter_output_values[layer_data_param.name] = layer

        # Generate layer preview
        try:
            layer_preview = generate_layer_preview(layer, tone_mapping_method=tone_mapping)
            layer_artifact = await self._upload_pil_image(
                layer_preview, f"{self.name}_{group_name}{_LAYER_PREVIEW_SUFFIX}.png"
            )
            self.parameter_output_values[layer_preview_param.name] = layer_artifact
        except Exception as e:
            logger.warning(
                "LoadEXR '%s': Layer preview for '%s' failed: %s",
                self.name,
                layer_display_name,
                e,
            )
            self.parameter_output_values[layer_preview_param.name] = None

        return layer_group

    def _show_selected_layer(self, selected_key: str) -> None:
        """Hide all layer groups except the one matching selected_key."""
        for key, group in self._layer_groups.items():
            is_selected = key == selected_key
            opts = dict(group.ui_options)
            opts["hide"] = not is_selected
            opts["collapsed"] = not is_selected
            group.ui_options = opts

    # --- Cleanup ---

    def _remove_dynamic_elements(self) -> None:
        """Remove all dynamically layer groups and part preview children."""
        self._layer_groups = {}
        self._clear_part_previews()
        elements_to_remove = [
            element
            for element in self.root_ui_element.find_elements_by_type(ParameterGroup)
            if element.name.startswith(_LAYER_PREFIX)
        ]
        for element in elements_to_remove:
            self.remove_parameter_element(element)

    # --- Image upload helper ---

    async def _upload_pil_image(self, image: Image.Image, filename: str) -> ImageUrlArtifact:
        """Upload a PIL Image to static storage and return an ImageUrlArtifact.

        Args:
            image: PIL Image to upload
            filename: Filename for the uploaded file

        Returns:
            ImageUrlArtifact with download URL

        Raises:
            RuntimeError: If upload or URL creation fails
        """
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_data = img_bytes.getvalue()

        # Get upload URL
        upload_request = CreateStaticFileUploadUrlRequest(file_name=filename)
        upload_result = GriptapeNodes.handle_request(upload_request)

        if isinstance(upload_result, CreateStaticFileUploadUrlResultFailure):
            msg = f"Failed to create upload URL for '{filename}': {upload_result.error}"
            raise RuntimeError(msg)
        if not isinstance(upload_result, CreateStaticFileUploadUrlResultSuccess):
            msg = f"Unexpected upload result type: {type(upload_result).__name__}"
            raise RuntimeError(msg)

        # Upload bytes asynchronously
        async with httpx.AsyncClient() as client:
            response = await client.request(
                upload_result.method,
                upload_result.url,
                content=img_data,
                headers=upload_result.headers,
                timeout=60,
            )
            response.raise_for_status()

        # Get download URL
        download_request = CreateStaticFileDownloadUrlRequest(file_name=filename)
        download_result = GriptapeNodes.handle_request(download_request)

        if isinstance(download_result, CreateStaticFileDownloadUrlResultFailure):
            msg = f"Failed to create download URL for '{filename}': {download_result.error}"
            raise RuntimeError(msg)
        if not isinstance(download_result, CreateStaticFileDownloadUrlResultSuccess):
            msg = f"Unexpected download result type: {type(download_result).__name__}"
            raise RuntimeError(msg)

        return ImageUrlArtifact(value=download_result.url)
