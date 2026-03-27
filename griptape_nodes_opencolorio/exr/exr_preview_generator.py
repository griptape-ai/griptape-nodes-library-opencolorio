"""EXR preview generators — BaseArtifactPreviewGenerator subclasses for OpenEXR files.

Two generators with distinct purposes:
- EXRPreviewGenerator: Tone-mapped sRGB preview of a layer or composite (RGB output)
- EXRChannelPreviewGenerator: Single-channel grayscale visualization (depth, alpha, mattes)

Both support part selection and layer selection within that part.
Callable directly by nodes now; designed to slot into the artifact manager later.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import PositiveInt

from griptape_nodes.retained_mode.events.os_events import (
    ExistingFilePolicy,
    WriteFileRequest,
    WriteFileResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.artifact_providers.base_artifact_preview_generator import (
    BaseArtifactPreviewGenerator,
)
from griptape_nodes.retained_mode.managers.artifact_providers.base_generator_parameters import (
    BaseGeneratorParameters,
    Field,
)

from griptape_nodes_opencolorio.exr.exr import (
    EXRChannelInfo,
    EXRData,
    EXRPart,
    ToneMappingMethod,
    apply_exposure,
    apply_gamma,
    generate_preview,
    load_layer_pixels,
    normalize_channel_role,
    parse_channel_name,
    scan_exr,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_part_index(exr_data: EXRData, part_index: int, file_path: str) -> EXRPart:
    """Get the part at the given index, raising ValueError if out of range."""
    if part_index < 0 or part_index >= len(exr_data.parts):
        msg = (
            f"Part index {part_index} is out of range for '{file_path}' "
            f"which has {len(exr_data.parts)} part(s)"
        )
        raise ValueError(msg)
    return exr_data.parts[part_index]


def _resolve_layer_channels(part: EXRPart, layer_name: str, file_path: str) -> list[EXRChannelInfo]:
    """Find a named layer within a part, returning its channels.

    Raises:
        ValueError: If the layer is not found in the part
    """
    for layer in part.layers:
        if layer.name == layer_name:
            return layer.channels

    available: list[str] = []
    for layer in part.layers:
        available.append(layer.name or "(default)")  # noqa: PERF401
    msg = (
        f"Layer '{layer_name}' not found in part {part.index} of '{file_path}'. "
        f"Available layers: {', '.join(available)}"
    )
    raise ValueError(msg)


def _resolve_default_channels(part: EXRPart, file_path: str) -> list[EXRChannelInfo]:
    """Find top-level RGBA channels in a part, falling back to first layer.

    Raises:
        ValueError: If no suitable channels are found
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
        return rgba_channels

    if part.layers:
        return part.layers[0].channels

    msg = f"No RGBA or layer channels found in part {part.index} of '{file_path}'"
    raise ValueError(msg)


def _resolve_channels(
    exr_data: EXRData, part_index: int, layer_name: str | None, file_path: str
) -> tuple[int, list[EXRChannelInfo]]:
    """Resolve which part and channels to load.

    Returns:
        Tuple of (part_index, channels_to_load)

    Raises:
        ValueError: If part index is out of range, layer not found, or no channels
    """
    part = _validate_part_index(exr_data, part_index, file_path)

    if layer_name is not None:
        channels = _resolve_layer_channels(part, layer_name, file_path)
    else:
        channels = _resolve_default_channels(part, file_path)

    return part.index, channels


def _parse_tone_mapping(value: str) -> ToneMappingMethod:
    """Convert a string tone mapping name to the enum value.

    Falls back to SIMPLE for unrecognized values.
    """
    try:
        return ToneMappingMethod(value.lower())
    except ValueError:
        return ToneMappingMethod.SIMPLE


def _write_preview(directory: str, filename: str, image_bytes: bytes) -> str:
    """Write preview bytes to disk via WriteFileRequest.

    Returns:
        The filename written.

    Raises:
        OSError: If writing fails
    """
    destination_path = str(Path(directory) / filename)

    write_request = WriteFileRequest(
        file_path=destination_path,
        content=image_bytes,
        create_parents=True,
        existing_file_policy=ExistingFilePolicy.OVERWRITE,
    )
    write_result = GriptapeNodes.handle_request(write_request)

    if not isinstance(write_result, WriteFileResultSuccess):
        msg = f"Failed to write EXR preview: {write_result.result_details}"
        raise OSError(msg)

    return filename


# ---------------------------------------------------------------------------
# EXR Preview Generator (RGB tone-mapped)
# ---------------------------------------------------------------------------


class EXRPreviewParameters(BaseGeneratorParameters):
    """Parameters for EXR RGB preview generation."""

    part_index: int = Field(
        default=0,
        description="Part index (0-based). Default 0 uses the first part.",
        editor_schema_type="integer",
        ge=0,
    )
    layer_name: str | None = Field(
        default=None,
        description="Layer name to render. None renders the default composite (top-level RGBA or first layer).",
        editor_schema_type="string",
    )
    tone_mapping: str = Field(
        default="simple",
        description="Tone mapping method: simple, reinhard, or filmic",
        editor_schema_type="string",
    )
    max_width: PositiveInt = Field(
        default=1024,
        description="Maximum width in pixels for generated preview (1-8192)",
        editor_schema_type="integer",
        le=8192,
    )
    max_height: PositiveInt = Field(
        default=1024,
        description="Maximum height in pixels for generated preview (1-8192)",
        editor_schema_type="integer",
        le=8192,
    )


class EXRPreviewGenerator(BaseArtifactPreviewGenerator):
    """EXR preview generator with tone mapping and optional part/layer selection.

    Generates tone-mapped sRGB preview images from OpenEXR files.
    Supports rendering a specific part, a named layer within that part,
    or the default composite (top-level RGBA / first layer of part 0).
    """

    def __init__(
        self,
        source_file_location: str,
        preview_format: str,
        destination_preview_directory: str,
        destination_preview_file_name: str,
        params: dict[str, Any],
    ) -> None:
        super().__init__(
            source_file_location,
            preview_format,
            destination_preview_directory,
            destination_preview_file_name,
            params,
        )
        self.params = EXRPreviewParameters.model_validate(params)

    @classmethod
    def get_friendly_name(cls) -> str:
        return "EXR Preview Generation"

    @classmethod
    def get_supported_source_formats(cls) -> set[str]:
        return {"exr"}

    @classmethod
    def get_supported_preview_formats(cls) -> set[str]:
        return {"png", "jpg", "webp"}

    @classmethod
    def get_parameters(cls) -> type[BaseGeneratorParameters]:
        return EXRPreviewParameters

    async def attempt_generate_preview(self) -> str:
        """Generate a tone-mapped sRGB preview from the EXR file.

        Raises:
            ValueError: If part index is out of range, layer not found, or no channels
            OSError: If writing the preview file fails
        """
        exr_data = scan_exr(self.source_file_location)

        part_index, channels = _resolve_channels(
            exr_data, self.params.part_index, self.params.layer_name, self.source_file_location
        )

        pixel_data = load_layer_pixels(self.source_file_location, part_index, channels)

        tone_mapping_method = _parse_tone_mapping(self.params.tone_mapping)

        preview_image = generate_preview(
            pixel_data,
            max_width=self.params.max_width,
            max_height=self.params.max_height,
            tone_mapping_method=tone_mapping_method,
        )

        output_buffer = BytesIO()
        preview_image.save(output_buffer, format=self.preview_format.upper())

        return _write_preview(
            self.destination_preview_directory,
            self.destination_preview_file_name,
            output_buffer.getvalue(),
        )


# ---------------------------------------------------------------------------
# EXR Channel Preview Generator (grayscale)
# ---------------------------------------------------------------------------


class EXRChannelPreviewParameters(BaseGeneratorParameters):
    """Parameters for EXR single-channel grayscale preview generation."""

    part_index: int = Field(
        default=0,
        description="Part index (0-based). Default 0 uses the first part.",
        editor_schema_type="integer",
        ge=0,
    )
    layer_name: str | None = Field(
        default=None,
        description="Layer containing the channel. None uses the default layer.",
        editor_schema_type="string",
    )
    channel_name: str = Field(
        default="R",
        description="Channel name to visualize (e.g., R, G, B, A, Z)",
        editor_schema_type="string",
    )
    normalize: bool = Field(
        default=False,
        description="Remap min/max to 0-1 range (useful for depth passes)",
        editor_schema_type="boolean",
    )
    exposure: float = Field(
        default=0.0,
        description="Exposure adjustment in stops",
        editor_schema_type="number",
    )
    gamma: float = Field(
        default=2.2,
        description="Gamma correction value",
        editor_schema_type="number",
        gt=0.0,
    )
    max_width: PositiveInt = Field(
        default=1024,
        description="Maximum width in pixels for generated preview (1-8192)",
        editor_schema_type="integer",
        le=8192,
    )
    max_height: PositiveInt = Field(
        default=1024,
        description="Maximum height in pixels for generated preview (1-8192)",
        editor_schema_type="integer",
        le=8192,
    )


class EXRChannelPreviewGenerator(BaseArtifactPreviewGenerator):
    """EXR single-channel preview generator for grayscale visualization.

    Extracts a single channel from an EXR layer and renders it as a
    grayscale image. Supports normalization (useful for depth passes),
    exposure, and gamma controls.
    """

    def __init__(
        self,
        source_file_location: str,
        preview_format: str,
        destination_preview_directory: str,
        destination_preview_file_name: str,
        params: dict[str, Any],
    ) -> None:
        super().__init__(
            source_file_location,
            preview_format,
            destination_preview_directory,
            destination_preview_file_name,
            params,
        )
        self.params = EXRChannelPreviewParameters.model_validate(params)

    @classmethod
    def get_friendly_name(cls) -> str:
        return "EXR Channel Preview Generation"

    @classmethod
    def get_supported_source_formats(cls) -> set[str]:
        return {"exr"}

    @classmethod
    def get_supported_preview_formats(cls) -> set[str]:
        return {"png", "jpg", "webp"}

    @classmethod
    def get_parameters(cls) -> type[BaseGeneratorParameters]:
        return EXRChannelPreviewParameters

    async def attempt_generate_preview(self) -> str:
        """Generate a grayscale preview of a single EXR channel.

        Raises:
            ValueError: If part/layer/channel not found
            OSError: If writing the preview file fails
        """
        exr_data = scan_exr(self.source_file_location)

        part_index, channels = _resolve_channels(
            exr_data, self.params.part_index, self.params.layer_name, self.source_file_location
        )

        channel_info = self._find_channel(channels, self.params.channel_name)

        pixel_data = load_layer_pixels(self.source_file_location, part_index, [channel_info])
        pixels = pixel_data[0].pixels

        if self.params.normalize:
            pixels = _normalize_pixels(pixels)

        if self.params.exposure != 0.0:
            pixels = apply_exposure(pixels, self.params.exposure)
        pixels = apply_gamma(pixels, self.params.gamma)

        pixels = np.clip(pixels, 0.0, 1.0)
        pixels_8bit = (pixels * 255.0).astype(np.uint8)
        grayscale_image = Image.fromarray(pixels_8bit, mode="L")

        grayscale_image.thumbnail(
            (self.params.max_width, self.params.max_height), Image.Resampling.LANCZOS
        )

        output_buffer = BytesIO()
        grayscale_image.save(output_buffer, format=self.preview_format.upper())

        return _write_preview(
            self.destination_preview_directory,
            self.destination_preview_file_name,
            output_buffer.getvalue(),
        )

    # --- Private instance methods ---

    def _find_channel(self, channels: list[EXRChannelInfo], channel_name: str) -> EXRChannelInfo:
        """Find a channel by its parsed short name (e.g., "R", "G", "Z").

        Raises:
            ValueError: If the channel is not found
        """
        for ch in channels:
            parsed = parse_channel_name(ch.name)
            if parsed.channel_name == channel_name:
                return ch

        available: list[str] = []
        for ch in channels:
            available.append(parse_channel_name(ch.name).channel_name)  # noqa: PERF401
        msg = (
            f"Channel '{channel_name}' not found. "
            f"Available channels: {', '.join(available)}"
        )
        raise ValueError(msg)


def _normalize_pixels(pixels: np.ndarray) -> np.ndarray:
    """Remap pixel values to 0-1 range based on actual min/max.

    Handles constant-value channels (where max == min) by returning zeros.
    """
    pmin = float(np.min(pixels))
    pmax = float(np.max(pixels))

    if pmax == pmin:
        return np.zeros_like(pixels)

    return (pixels - pmin) / (pmax - pmin)
