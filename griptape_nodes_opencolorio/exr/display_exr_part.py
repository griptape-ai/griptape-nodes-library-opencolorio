"""Display EXR Part node — tone-mapped composite preview of a single EXR part."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

from griptape_nodes_opencolorio.exr.exr import (
    ToneMappingMethod,
    generate_preview,
    load_layer_pixels,
    parse_channel_name,
    scan_exr,
)
from griptape_nodes_opencolorio.exr.exr_layer_artifact import EXRLayerArtifact
from griptape_nodes_opencolorio.exr.exr_part_artifact import EXRPartArtifact
from griptape_nodes_opencolorio.exr.exr_preview_generator import (
    _parse_tone_mapping,
    _resolve_default_channels,
    _validate_part_index,
)

logger = logging.getLogger("griptape_nodes")

_LAYER_PREFIX = "layer_"
_DEFAULT_LAYER_NAME = "default"


class DisplayEXRPart(ControlNode):
    """Display a tone-mapped composite preview of a single EXR part.

    Takes an EXRPartArtifact, renders a Nuke-style composite preview
    (top-level RGBA or first layer), and exposes per-layer artifacts
    for downstream nodes.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # --- Inputs ---

        self._part_param = Parameter(
            name="part",
            type="EXRPartArtifact",
            input_types=["EXRPartArtifact"],
            tooltip="EXR part descriptor to display",
            allowed_modes={ParameterMode.INPUT},
        )
        self.add_parameter(self._part_param)

        self._tone_mapping_param = ParameterString(
            name="tone_mapping",
            default_value=ToneMappingMethod.SIMPLE.value,
            tooltip="Tone mapping method for sRGB preview",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self._tone_mapping_param.add_trait(Options(choices=[m.value for m in ToneMappingMethod]))
        self.add_parameter(self._tone_mapping_param)

        # --- Preview output ---

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="display_exr_part.png",
        )
        self._output_file.add_parameter()

        self._output_param = ParameterImage(
            name="output",
            display_name="sRGB Preview Image",
            default_value=None,
            tooltip="Tone-mapped sRGB preview of the part's default channels",
            allowed_modes={ParameterMode.OUTPUT},
            ui_options={"pulse_on_run": True},
        )
        self.add_parameter(self._output_param)

        # --- Metadata outputs ---

        self._part_index_param = ParameterInt(
            name="part_index",
            default_value=0,
            tooltip="Part index (0-based)",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._part_index_param)

        self._name_param = ParameterString(
            name="name",
            default_value="",
            tooltip="Part name (empty for unnamed parts)",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._name_param)

        self._width_param = ParameterInt(
            name="width",
            default_value=0,
            tooltip="Image width in pixels",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._width_param)

        self._height_param = ParameterInt(
            name="height",
            default_value=0,
            tooltip="Image height in pixels",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._height_param)

        # --- Dynamic layers group ---

        self._layers_group = ParameterGroup(name="exr_layers")
        self._layers_group.ui_options = {"display_name": "Layers"}
        self.add_node_element(self._layers_group)

    # --- Public instance methods ---

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "part" and isinstance(value, EXRPartArtifact):
            self._update_metadata(value)
            self._populate_layers(value)
        return super().after_value_set(parameter, value)

    async def aprocess(self) -> None:
        part_artifact = self.get_parameter_value("part")
        if not isinstance(part_artifact, EXRPartArtifact):
            return

        self._update_metadata(part_artifact)
        self._populate_layers(part_artifact)

        tone_mapping_str = self.get_parameter_value("tone_mapping") or ToneMappingMethod.SIMPLE.value
        tone_mapping = _parse_tone_mapping(tone_mapping_str)

        file_path = part_artifact.file_path
        exr_data = scan_exr(file_path)
        part = _validate_part_index(exr_data, part_artifact.part_index, file_path)
        channels = _resolve_default_channels(part, file_path)

        pixel_data = load_layer_pixels(file_path, part.index, channels)
        preview = generate_preview(pixel_data, tone_mapping_method=tone_mapping)

        img_bytes = BytesIO()
        preview.save(img_bytes, format="PNG")
        dest = self._output_file.build_file()
        saved = dest.write_bytes(img_bytes.getvalue())
        artifact = ImageUrlArtifact(value=saved.location)
        self.publish_update_to_parameter("output", artifact)
        self.parameter_output_values["output"] = artifact

    # --- Private instance methods ---

    def _update_metadata(self, part_artifact: EXRPartArtifact) -> None:
        """Set metadata output values from the part artifact."""
        self.parameter_output_values["part_index"] = part_artifact.part_index
        self.parameter_output_values["name"] = part_artifact.name
        self.parameter_output_values["width"] = part_artifact.width
        self.parameter_output_values["height"] = part_artifact.height

    def _populate_layers(self, part_artifact: EXRPartArtifact) -> None:
        """Clear and repopulate the layers group from the part's layers."""
        for child in list(self._layers_group.children):
            self._layers_group.remove_child(child)

        for layer in part_artifact.layers:
            layer_artifact = EXRLayerArtifact(part=part_artifact, layer=layer)
            layer_display = layer.name or _DEFAULT_LAYER_NAME
            channel_names: list[str] = []
            for ch in layer.channels:
                channel_names.append(parse_channel_name(ch.name).channel_name)  # noqa: PERF401

            channels_str = ", ".join(channel_names)

            layer_param = Parameter(
                name=f"{_LAYER_PREFIX}{layer.name or _DEFAULT_LAYER_NAME}",
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
