"""LoadEXR node for loading OpenEXR files and exposing layers.

Architecture: Two-phase lazy loading via OIIO.
- Phase 1 (after_value_set on file path): Scan headers only (no pixels).
  Populates UI, generates composite preview (loads just RGBA channels).
- Phase 2 (after_value_set on layer select): Loads just that layer's RGB
  channels for preview. Outputs EXRLayerArtifact descriptors (no pixels).
- Downstream nodes use artifacts to load exactly the channels they need.
"""

from __future__ import annotations

import json
import logging
from io import BytesIO
from typing import Any

import httpx
from griptape.artifacts import ImageUrlArtifact
from PIL import Image

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

from griptape_nodes_opencolorio.exr.exr import (
    EXRChannelInfo,
    EXRChannelPixelData,
    EXRData,
    EXRLayer,
    EXRPart,
    ToneMappingMethod,
    generate_preview,
    load_layer_pixels,
    normalize_channel_role,
    parse_channel_name,
    scan_exr,
)
from griptape_nodes_opencolorio.exr.exr_file_artifact import EXRFileArtifact
from griptape_nodes_opencolorio.exr.exr_layer_artifact import EXRLayerArtifact
from griptape_nodes_opencolorio.exr.exr_part_artifact import EXRPartArtifact

logger = logging.getLogger("griptape_nodes")



# Naming constants for dynamically created elements
_LAYER_PREFIX = "layer_"
_PART_PREFIX = "part_"
_LAYER_PREVIEW_SUFFIX = "_preview"
_LAYER_CHANNELS_SUFFIX = "_channels"
_LAYER_DATA_SUFFIX = "_data"
_PART_PREVIEW_SUFFIX = "_composite"
_DEFAULT_LAYER_NAME = "default"


def _part_prefix(part: EXRPart, exr_data: EXRData) -> str:
    """Return a disambiguating prefix for multi-part EXR files, empty for single-part."""
    if len(exr_data.parts) > 1:
        return f"p{part.index}_"
    return ""


def _layer_key(layer: EXRLayer, part_prefix: str) -> str:
    """Build the canonical key used to identify a layer in the selector and group names."""
    return f"{part_prefix}{layer.name or _DEFAULT_LAYER_NAME}"



class LoadEXR(SuccessFailureNode):
    """Load an OpenEXR file and expose its layers, channels, and metadata.

    Outputs EXRFileArtifact and EXRLayerArtifact descriptors — downstream
    nodes use these to load only the pixel data they need.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self._cached_exr_data: EXRData | None = None
        self._cached_file_path: str = ""
        self._cached_tone_mapping = ToneMappingMethod.SIMPLE
        self._layer_groups: dict[str, ParameterGroup] = {}
        self._layer_lookup: dict[str, tuple[int, EXRLayer]] = {}

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

        self._tone_mapping_param = ParameterString(
            name="tone_mapping",
            default_value=ToneMappingMethod.SIMPLE.value,
            tooltip="Tone mapping method for sRGB preview generation",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=[m.value for m in ToneMappingMethod])},
        )
        self.add_parameter(self._tone_mapping_param)

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

        # EXR Part Preview group (children populated dynamically after scan)
        self._part_preview_group = ParameterGroup(name="exr_part_preview")
        self._part_preview_group.ui_options = {"display_name": "EXR Part Previews"}
        self.add_node_element(self._part_preview_group)

        # Layer selector (populated after scan)
        self._view_layer_param = ParameterString(
            name="view_layer",
            display_name="Select Layer",
            default_value="",
            tooltip="Choose which EXR layer to display",
            allowed_modes={ParameterMode.PROPERTY},
            traits={Options(choices=[])},
        )
        self.add_parameter(self._view_layer_param)

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

        self._selected_layer_param = Parameter(
            name="selected_layer",
            display_name="Selected Layer",
            type="EXRLayerArtifact",
            output_type="EXRLayerArtifact",
            tooltip="Descriptor for the currently selected layer",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._selected_layer_param)

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

        self._create_status_parameters(
            result_details_tooltip="Details about the EXR load result",
            result_details_placeholder="Load details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    # --- Public instance methods ---

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter is self._file_path_param:
            self._on_file_path_changed(str(value) if value else "")
        elif parameter is self._view_layer_param:
            self._on_layer_selected(str(value) if value else "")

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
        layer_names = [
            layer.name or _DEFAULT_LAYER_NAME
            for p in exr_data.parts
            for layer in p.layers
        ]
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
        self._cached_file_path = ""
        self._layer_lookup = {}

        if not file_path:
            return

        try:
            exr_data = scan_exr(file_path)
        except (ValueError, RuntimeError) as e:
            logger.error(
                "LoadEXR '%s': Attempted to scan EXR file '%s'. Failed because: %s",
                self.name, file_path, e,
            )
            return

        self._cached_exr_data = exr_data
        self._cached_file_path = file_path
        self._cached_tone_mapping = ToneMappingMethod(
            self.get_parameter_value(self._tone_mapping_param.name) or ToneMappingMethod.SIMPLE.value
        )

        part = exr_data.parts[0]

        self.parameter_output_values[self._image_width_param.name] = part.width
        self.parameter_output_values[self._image_height_param.name] = part.height
        self._populate_exr_info(exr_data)

        file_artifact = self._build_file_artifact(file_path, exr_data)
        self.parameter_output_values[self._exr_file_param.name] = file_artifact

        self._load_and_show_composite_preview(file_path, exr_data)
        self._populate_layers(exr_data)

    def _on_layer_selected(self, selected_key: str) -> None:
        """Show the selected layer group and generate its preview."""
        if not selected_key or not self._cached_file_path:
            self._show_selected_layer(selected_key)
            return

        lookup = self._layer_lookup.get(selected_key)
        if not lookup:
            self._show_selected_layer(selected_key)
            return

        part_index, layer = lookup

        self._show_selected_layer(selected_key)

        layer_artifact = self._build_layer_artifact(
            self._cached_file_path, part_index, layer
        )
        self.parameter_output_values[self._selected_layer_param.name] = layer_artifact

        self._load_and_show_layer_preview(
            self._cached_file_path, part_index, layer, selected_key
        )

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

        self.parameter_output_values[self._custom_attributes_param.name] = (
            json.dumps(header.custom, indent=2, default=str) if header.custom else "{}"
        )

    def _load_and_show_composite_preview(self, file_path: str, exr_data: EXRData) -> None:
        """Load just RGBA channels for composite preview, generate thumbnail."""
        self._clear_part_previews()

        is_multi_part = len(exr_data.parts) > 1
        if is_multi_part:
            display_name = "EXR Part Previews"
        else:
            display_name = "EXR Part Preview"
        self._part_preview_group.ui_options = {"display_name": display_name}

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
                pixel_data = self._load_composite_channels(file_path, part)
                preview_image = generate_preview(
                    pixel_data, tone_mapping_method=self._cached_tone_mapping
                )
                preview_artifact = self._upload_pil_image_sync(
                    preview_image, f"{self.name}_{_PART_PREFIX}{part.index}_preview.png"
                )
                self.parameter_output_values[preview_param.name] = preview_artifact
            except (ValueError, RuntimeError) as e:
                logger.warning(
                    "LoadEXR '%s': Composite preview for part %d failed: %s",
                    self.name, part.index, e,
                )
                self.parameter_output_values[preview_param.name] = None

    def _load_composite_channels(self, file_path: str, part: EXRPart) -> list[EXRChannelPixelData]:
        """Load top-level RGBA channels (or first layer's channels) for composite preview.

        Returns:
            Loaded pixel data for the selected channels.

        Raises:
            ValueError: If no suitable channels are found for composite preview.
        """
        rgba_channels: list[EXRChannelInfo] = []
        for ch in part.channels:
            parsed = parse_channel_name(ch.name)
            if parsed.layer_name != "":
                continue
            role = normalize_channel_role(parsed.channel_name)
            if role in ("red", "green", "blue", "alpha"):
                rgba_channels.append(ch)

        if rgba_channels:
            channels_to_load = rgba_channels
        elif part.layers:
            channels_to_load = part.layers[0].channels
        else:
            msg = f"LoadEXR '{self.name}': Attempted to load composite channels for part {part.index}. Failed because no RGBA or layer channels were found."
            raise ValueError(msg)

        return load_layer_pixels(file_path, part.index, channels_to_load)

    def _populate_layers(self, exr_data: EXRData) -> None:
        """Create all layer groups (hidden) and set up the layer selector."""
        self._layer_groups = {}
        self._layer_lookup = {}
        layer_choices: list[str] = []

        for part in exr_data.parts:
            prefix = _part_prefix(part, exr_data)
            for layer in part.layers:
                key = _layer_key(layer, prefix)
                layer_choices.append(key)
                group = self._create_layer_group(layer, prefix)
                self._layer_groups[key] = group
                self._layer_lookup[key] = (part.index, layer)

        existing_traits = self._view_layer_param.find_elements_by_type(Options)
        for trait in existing_traits:
            self._view_layer_param.remove_trait(trait_type=trait)
        self._view_layer_param.add_trait(Options(choices=layer_choices))

        if layer_choices:
            default_layer = layer_choices[0]
        else:
            default_layer = ""
        self.set_parameter_value(self._view_layer_param.name, default_layer)
        self._on_layer_selected(default_layer)

    def _create_layer_group(
        self,
        layer: EXRLayer,
        part_prefix: str,
    ) -> ParameterGroup:
        """Create a ParameterGroup for a single EXR layer (no pixel loading)."""
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

        layer_preview_param = ParameterImage(
            name=f"{group_name}{_LAYER_PREVIEW_SUFFIX}",
            display_name="Layer Preview",
            allowed_modes={ParameterMode.OUTPUT},
            tooltip=f"sRGB preview of layer '{layer_display_name}'",
            settable=False,
        )
        layer_group.add_child(layer_preview_param)

        channels_param = Parameter(
            name=f"{group_name}{_LAYER_CHANNELS_SUFFIX}",
            display_name="Channels",
            type="list",
            default_value=channel_names,
            tooltip=f"Channel names in layer '{layer_display_name}'",
            allowed_modes={ParameterMode.OUTPUT},
        )
        layer_group.add_child(channels_param)

        layer_data_param = Parameter(
            name=f"{group_name}{_LAYER_DATA_SUFFIX}",
            display_name="Layer Data",
            type="EXRLayerArtifact",
            output_type="EXRLayerArtifact",
            tooltip=f"Descriptor for layer '{layer_display_name}' (use to load pixels on demand)",
            allowed_modes={ParameterMode.OUTPUT},
        )
        layer_group.add_child(layer_data_param)

        self.add_node_element(layer_group)
        self.parameter_output_values[channels_param.name] = channel_names

        return layer_group

    def _load_and_show_layer_preview(
        self,
        file_path: str,
        part_index: int,
        layer: EXRLayer,
        layer_key: str,
    ) -> None:
        """Load a layer's RGB channels and generate its preview thumbnail."""
        group = self._layer_groups.get(layer_key)
        if not group:
            return

        group_name = group.name
        preview_param_name = f"{group_name}{_LAYER_PREVIEW_SUFFIX}"
        data_param_name = f"{group_name}{_LAYER_DATA_SUFFIX}"

        layer_artifact = self._build_layer_artifact(file_path, part_index, layer)
        self.parameter_output_values[data_param_name] = layer_artifact

        try:
            pixel_data = load_layer_pixels(file_path, part_index, layer.channels)
            layer_preview = generate_preview(
                pixel_data, tone_mapping_method=self._cached_tone_mapping
            )
            preview_artifact = self._upload_pil_image_sync(
                layer_preview, f"{self.name}_{group_name}{_LAYER_PREVIEW_SUFFIX}.png"
            )
            self.parameter_output_values[preview_param_name] = preview_artifact
        except (ValueError, RuntimeError) as e:
            logger.warning(
                "LoadEXR '%s': Layer preview for '%s' failed: %s",
                self.name, layer.name or _DEFAULT_LAYER_NAME, e,
            )
            self.parameter_output_values[preview_param_name] = None

    def _show_selected_layer(self, selected_key: str) -> None:
        """Hide all layer groups except the one matching selected_key."""
        for key, group in self._layer_groups.items():
            is_selected = key == selected_key
            opts = dict(group.ui_options)
            opts["hide"] = not is_selected
            opts["collapsed"] = not is_selected
            group.ui_options = opts

    def _build_file_artifact(self, file_path: str, exr_data: EXRData) -> EXRFileArtifact:
        """Build an EXRFileArtifact from scanned EXR data."""
        parts: list[EXRPartArtifact] = []
        for part in exr_data.parts:
            part_artifact = self._build_part_artifact(file_path, part.index)
            parts.append(part_artifact)
        return EXRFileArtifact(file_path=file_path, parts=parts)

    def _build_layer_artifact(
        self, file_path: str, part_index: int, layer: EXRLayer
    ) -> EXRLayerArtifact:
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

    def _clear_part_previews(self) -> None:
        """Remove all dynamically created preview children from the part preview group."""
        children = list(self._part_preview_group.find_elements_by_type(ParameterImage))
        for child in children:
            self._part_preview_group.remove_child(child)

    def _remove_dynamic_elements(self) -> None:
        """Remove all dynamic layer groups and part preview children."""
        self._layer_groups = {}
        self._layer_lookup = {}
        self._clear_part_previews()
        elements_to_remove = [
            element
            for element in self.root_ui_element.find_elements_by_type(ParameterGroup)
            if element.name.startswith(_LAYER_PREFIX)
        ]
        for element in elements_to_remove:
            self.remove_parameter_element(element)

    def _upload_pil_image_sync(self, image: Image.Image, filename: str) -> ImageUrlArtifact:
        """Upload a PIL Image to static storage and return an ImageUrlArtifact.

        Synchronous version for use in after_value_set callbacks.

        Raises:
            RuntimeError: If upload, HTTP request, or URL creation fails
        """
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_data = img_bytes.getvalue()

        upload_request = CreateStaticFileUploadUrlRequest(file_name=filename)
        upload_result = GriptapeNodes.handle_request(upload_request)

        if isinstance(upload_result, CreateStaticFileUploadUrlResultFailure):
            msg = f"LoadEXR '{self.name}': Attempted to upload preview '{filename}'. Failed because: {upload_result.error}"
            raise RuntimeError(msg)
        if not isinstance(upload_result, CreateStaticFileUploadUrlResultSuccess):
            msg = f"LoadEXR '{self.name}': Attempted to upload preview '{filename}'. Failed with unexpected result type: {type(upload_result).__name__}"
            raise RuntimeError(msg)

        response = httpx.request(
            upload_result.method,
            upload_result.url,
            content=img_data,
            headers=upload_result.headers,
            timeout=60,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            msg = f"LoadEXR '{self.name}': Attempted to upload preview '{filename}'. Failed with HTTP {response.status_code}."
            raise RuntimeError(msg) from e

        download_request = CreateStaticFileDownloadUrlRequest(file_name=filename)
        download_result = GriptapeNodes.handle_request(download_request)

        if isinstance(download_result, CreateStaticFileDownloadUrlResultFailure):
            msg = f"LoadEXR '{self.name}': Attempted to get download URL for '{filename}'. Failed because: {download_result.error}"
            raise RuntimeError(msg)
        if not isinstance(download_result, CreateStaticFileDownloadUrlResultSuccess):
            msg = f"LoadEXR '{self.name}': Attempted to get download URL for '{filename}'. Failed with unexpected result type: {type(download_result).__name__}"
            raise RuntimeError(msg)

        return ImageUrlArtifact(value=download_result.url)
