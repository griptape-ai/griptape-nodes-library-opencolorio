"""EXR data structures, I/O operations, and tone mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, NamedTuple

import numpy as np
import OpenEXR  # type: ignore[import-not-found]
from PIL import Image


class CompressionType(StrEnum):
    """OpenEXR compression types (matches OpenEXR.Compression enum names)."""

    NO_COMPRESSION = "NO_COMPRESSION"
    RLE_COMPRESSION = "RLE_COMPRESSION"
    ZIPS_COMPRESSION = "ZIPS_COMPRESSION"
    ZIP_COMPRESSION = "ZIP_COMPRESSION"
    PIZ_COMPRESSION = "PIZ_COMPRESSION"
    PXR24_COMPRESSION = "PXR24_COMPRESSION"
    B44_COMPRESSION = "B44_COMPRESSION"
    B44A_COMPRESSION = "B44A_COMPRESSION"
    DWAA_COMPRESSION = "DWAA_COMPRESSION"
    DWAB_COMPRESSION = "DWAB_COMPRESSION"
    HTJ2K32_COMPRESSION = "HTJ2K32_COMPRESSION"
    HTJ2K256_COMPRESSION = "HTJ2K256_COMPRESSION"


class LineOrderType(StrEnum):
    """OpenEXR line order types (matches OpenEXR.LineOrder enum names)."""

    INCREASING_Y = "INCREASING_Y"
    DECREASING_Y = "DECREASING_Y"
    RANDOM_Y = "RANDOM_Y"


class StorageType(StrEnum):
    """OpenEXR storage types (matches OpenEXR.Storage enum names)."""

    SCANLINE_IMAGE = "scanlineimage"
    TILED_IMAGE = "tiledimage"


class WindowCoordinates(NamedTuple):
    """EXR window coordinates (min and max corners)."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int


class ChannelNameParts(NamedTuple):
    """Parsed components of an EXR channel name.

    Attributes:
        layer_name: Layer prefix (empty string for default layer)
        channel_name: Channel component (R, G, B, A, Z, etc.)
    """

    layer_name: str
    channel_name: str


@dataclass
class EXRChannel:
    """Channel data and metadata from an OpenEXR file.

    Attributes:
        name: Channel name (e.g., "R", "G", "B", "Z", "beauty.R")
        pixels: Pixel data as NumPy array (float32)
        x_sampling: Horizontal sampling rate (usually 1)
        y_sampling: Vertical sampling rate (usually 1)
    """

    name: str
    pixels: np.ndarray
    x_sampling: int
    y_sampling: int


@dataclass
class EXRLayer:
    """Layer grouping channels with same prefix.

    Attributes:
        name: Layer name (empty string for default/main layer)
        channels: List of channels in this layer

    Access patterns:
        # Get channel by name
        channels_by_name = {ch.name: ch for ch in layer.channels}
        r_channel = channels_by_name.get("R")

        # Check if layer has RGBA
        has_rgba = all(ch in channels_by_name for ch in ["R", "G", "B", "A"])
    """

    name: str
    channels: list[EXRChannel]


@dataclass
class EXRHeader:
    """Header metadata from an OpenEXR file part.

    Contains all required OpenEXR attributes plus optional/custom attributes.
    Required attributes per OpenEXR specification:
    - compression, line_order, data_window, display_window
    - pixel_aspect_ratio, screen_window_center, screen_window_width, storage_type

    Multi-part files also have: name, chunk_count
    """

    # Required attributes (always present in valid EXR)
    compression: CompressionType
    line_order: LineOrderType
    data_window: WindowCoordinates
    display_window: WindowCoordinates
    pixel_aspect_ratio: float
    screen_window_center: tuple[float, float]  # (x, y)
    screen_window_width: float
    storage_type: StorageType

    # Multi-part attributes (present only in multi-part files)
    name: str  # Part name (empty string for single-part)
    chunk_count: int | None  # Present in multi-part files

    # Custom/optional attributes
    custom: dict[str, Any]  # User-defined attributes (renderTime, owner, etc.)


@dataclass
class EXRPart:
    """Single part from an OpenEXR file.

    Multi-part EXR files have multiple parts, each with independent:
    - Channels (different channel sets per part)
    - Header (different compression, attributes per part)
    - Metadata (name, index)

    Single-part files have exactly one part.

    Access patterns:
        # Legacy flat access
        channels_by_name = {ch.name: ch for ch in part.channels}
        rgb_pixels = channels_by_name["R"].pixels

        # New layer-based access
        layers_by_name = {layer.name: layer for layer in part.layers}
        beauty_layer = layers_by_name.get("beauty")
        if beauty_layer:
            beauty_r = next(ch for ch in beauty_layer.channels if ch.name.endswith(".R"))

        # Default layer access
        default_layer = next(l for l in part.layers if l.name == "")
    """

    channels: list[EXRChannel]  # Flat list for backward compatibility
    layers: list[EXRLayer]  # Structured layer access
    header: EXRHeader
    index: int  # Part index (0-based)
    width: int  # From part.width
    height: int  # From part.height


@dataclass
class EXRData:
    """All data extracted from an OpenEXR file.

    Contains list of parts (1 for single-part, N for multi-part).

    Access patterns:
        Single-part: exr_data.parts[0].channels
        Multi-part: exr_data.parts[1].channels
        Find part by name: next(p for p in exr_data.parts if p.header.name == "beauty")
    """

    parts: list[EXRPart]


def parse_channel_name(full_name: str) -> ChannelNameParts:
    """Parse EXR channel name into layer and channel components.

    Args:
        full_name: Full channel name (e.g., "beauty.R", "R", "spec.indirect.G")

    Returns:
        ChannelNameParts with layer_name and channel_name
        - layer_name: Empty string for default layer, otherwise layer prefix
        - channel_name: The actual channel (R, G, B, A, Z, etc.)

    Examples:
        "R" → ChannelNameParts("", "R")
        "beauty.R" → ChannelNameParts("beauty", "R")
        "diffuse.indirect.R" → ChannelNameParts("diffuse_indirect", "R")
        "Ci.R" → ChannelNameParts("", "R")  # Ci is RenderMan default layer
    """
    # Split by period
    parts = full_name.split(".")

    if len(parts) == 1:
        # No prefix - default layer
        return ChannelNameParts(layer_name="", channel_name=parts[0])

    # Last part is channel name
    channel_name = parts[-1]

    # Everything before is layer name (joined with underscores)
    layer_parts = parts[:-1]
    layer_name = "_".join(layer_parts)

    # Special case: "Ci" is RenderMan's default layer
    if layer_name == "Ci":
        layer_name = ""

    return ChannelNameParts(layer_name=layer_name, channel_name=channel_name)


def group_channels_into_layers(channels: list[EXRChannel]) -> list[EXRLayer]:
    """Group channels by layer name.

    Args:
        channels: Flat list of channels from EXR part

    Returns:
        List of EXRLayer objects, sorted by layer name (default layer first)

    Example:
        Input channels: ["R", "G", "B", "beauty.R", "beauty.G", "Z"]
        Output layers:
        - Layer(name="", channels=["R", "G", "B", "Z"])
        - Layer(name="beauty", channels=["beauty.R", "beauty.G"])
    """
    # Group channels by layer name
    layers_dict: dict[str, list[EXRChannel]] = {}

    for channel in channels:
        parsed = parse_channel_name(channel.name)
        layer_name = parsed.layer_name

        if layer_name not in layers_dict:
            layers_dict[layer_name] = []
        layers_dict[layer_name].append(channel)

    # Convert to list of EXRLayer objects
    layers = [EXRLayer(name=layer_name, channels=channels_list) for layer_name, channels_list in layers_dict.items()]

    # Sort: default layer ("") first, then alphabetically
    layers.sort(key=lambda layer: ("" if layer.name == "" else f"~{layer.name}"))

    return layers


def attempt_read_exr(file_path: str) -> EXRData:  # noqa: C901, PLR0912, PLR0915 - Complex data extraction
    """Read OpenEXR file and extract ALL data with proper structure.

    Args:
        file_path: Path to the EXR file

    Returns:
        EXRData containing list of EXRPart objects with structured headers

    Raises:
        ValueError: If file_path is empty, EXR has no channels, or unknown enum values
        RuntimeError: If EXR loading fails (includes FileNotFoundError from OpenEXR)

    Note:
        Supports multi-part EXR files. Each part is returned as separate EXRPart.
    """
    # FAILURES FIRST
    if not file_path:
        msg = "file_path must not be empty"
        raise ValueError(msg)

    # Load EXR using OpenEXR 3.4.x with context manager for resource cleanup
    try:
        with OpenEXR.File(file_path) as exr:
            # Get all parts (1 for single-part, >1 for multi-part)
            parts_list = exr.parts  # Property, not method

            # Extract each part with all its data
            exr_parts = []
            for part in parts_list:
                # Get raw header dict and channels dict
                raw_header = part.header  # Property, not method
                raw_channels = part.channels  # Property, not method

                # FAILURES - No channels in this part
                if not raw_channels:
                    msg = f"EXR file part {part.part_index} has no channels: {file_path}"
                    raise ValueError(msg)  # noqa: TRY301 - Simple validation check

                # Build EXRChannel objects with metadata
                channels_list = []
                for ch_name, ch_obj in raw_channels.items():
                    # Find metadata for this channel from header's channel list
                    # Channel objects have string representation: Channel("name", xSampling=1, ySampling=1)
                    channel_name_prefix = f'Channel("{ch_name}"'
                    ch_metadata = None
                    for channel_obj in raw_header["channels"]:
                        if str(channel_obj).startswith(channel_name_prefix):
                            ch_metadata = channel_obj
                            break

                    x_sampling = 1  # Default
                    y_sampling = 1  # Default
                    if ch_metadata:
                        # Parse xSampling, ySampling from Channel object string representation
                        ch_str = str(ch_metadata)
                        if "xSampling=" in ch_str:
                            x_sampling = int(ch_str.split("xSampling=", maxsplit=1)[1].split(",", maxsplit=1)[0].split(")", maxsplit=1)[0])
                        if "ySampling=" in ch_str:
                            y_sampling = int(ch_str.split("ySampling=", maxsplit=1)[1].split(")", maxsplit=1)[0])

                    exr_channel = EXRChannel(
                        name=ch_name,
                        pixels=ch_obj.pixels.copy(),  # Copy before file closes
                        x_sampling=x_sampling,
                        y_sampling=y_sampling,
                    )
                    channels_list.append(exr_channel)

                # Extract required header attributes
                data_window_raw = raw_header["dataWindow"]
                display_window_raw = raw_header["displayWindow"]

                # Convert raw tuples to WindowCoordinates
                min_coords, max_coords = data_window_raw
                data_window = WindowCoordinates(
                    xmin=int(min_coords[0]),
                    ymin=int(min_coords[1]),
                    xmax=int(max_coords[0]),
                    ymax=int(max_coords[1]),
                )

                min_coords, max_coords = display_window_raw
                display_window = WindowCoordinates(
                    xmin=int(min_coords[0]),
                    ymin=int(min_coords[1]),
                    xmax=int(max_coords[0]),
                    ymax=int(max_coords[1]),
                )

                # Extract screen window center (numpy array -> tuple)
                screen_center = raw_header["screenWindowCenter"]
                screen_center_tuple = (float(screen_center[0]), float(screen_center[1]))

                # Convert OpenEXR enums to StrEnums (will raise ValueError if unknown)
                compression = CompressionType(raw_header["compression"].name)
                line_order = LineOrderType(raw_header["lineOrder"].name)
                storage_type = StorageType(raw_header["type"].name)

                # Separate required vs custom attributes
                required_attrs = {
                    "channels",
                    "compression",
                    "dataWindow",
                    "displayWindow",
                    "lineOrder",
                    "pixelAspectRatio",
                    "screenWindowCenter",
                    "screenWindowWidth",
                    "type",
                    "name",
                    "chunkCount",
                }
                custom_attrs = {k: v for k, v in raw_header.items() if k not in required_attrs}

                # Build EXRHeader
                exr_header = EXRHeader(
                    compression=compression,
                    line_order=line_order,
                    data_window=data_window,
                    display_window=display_window,
                    pixel_aspect_ratio=raw_header["pixelAspectRatio"],
                    screen_window_center=screen_center_tuple,
                    screen_window_width=raw_header["screenWindowWidth"],
                    storage_type=storage_type,
                    name=raw_header.get("name", ""),
                    chunk_count=raw_header.get("chunkCount", None),
                    custom=custom_attrs,
                )

                # Build EXRPart (using Part's built-in width/height/index)
                exr_part = EXRPart(
                    channels=channels_list,
                    layers=group_channels_into_layers(channels_list),
                    header=exr_header,
                    index=part.part_index,
                    width=part.width(),
                    height=part.height(),
                )
                exr_parts.append(exr_part)

            # FAILURES - No parts at all
            if not exr_parts:
                msg = f"EXR file has no parts: {file_path}"
                raise ValueError(msg)  # noqa: TRY301 - Simple validation check

            # Multi-part validation: Nuke requires consistent windows across parts.
            if len(exr_parts) > 1:
                _validate_multi_part_consistency(exr_parts)
                _apply_legacy_part_name_prefix(exr_parts)

            # SUCCESS PATH - Return everything extracted
            return EXRData(parts=exr_parts)

    except ValueError:
        raise
    except Exception as e:
        msg = f"Failed to load EXR file '{file_path}': {e}"
        raise RuntimeError(msg) from e


def _validate_multi_part_consistency(parts: list[EXRPart]) -> None:
    """Validate that all parts have consistent windows and metadata.

    Nuke (exrReader.cpp:1935-1943) requires multi-part EXRs to have the same
    data window, display window, pixel aspect ratio, and line order across all parts.

    Raises:
        ValueError: If any part differs from part 0.
    """
    ref = parts[0]
    for part in parts[1:]:
        if part.header.data_window != ref.header.data_window:
            msg = (
                f"Multi-part EXR has inconsistent data windows: "
                f"part 0={ref.header.data_window}, part {part.index}={part.header.data_window}"
            )
            raise ValueError(msg)
        if part.header.display_window != ref.header.display_window:
            msg = (
                f"Multi-part EXR has inconsistent display windows: "
                f"part 0={ref.header.display_window}, part {part.index}={part.header.display_window}"
            )
            raise ValueError(msg)
        if part.header.pixel_aspect_ratio != ref.header.pixel_aspect_ratio:
            msg = (
                f"Multi-part EXR has inconsistent pixel aspect ratios: "
                f"part 0={ref.header.pixel_aspect_ratio}, part {part.index}={part.header.pixel_aspect_ratio}"
            )
            raise ValueError(msg)
        if part.header.line_order != ref.header.line_order:
            msg = (
                f"Multi-part EXR has inconsistent line orders: "
                f"part 0={ref.header.line_order}, part {part.index}={part.header.line_order}"
            )
            raise ValueError(msg)


def _apply_legacy_part_name_prefix(parts: list[EXRPart]) -> None:
    """Detect and handle legacy multi-part files that use part names as layer names.

    Nuke (exrReader.cpp:1887-1924) auto-detects legacy multi-part files where the
    part name stores the layer name (channels are just R, G, B without '.' separators).
    If no channels in any part contain a '.' separator, the part's header name is
    prepended as a layer prefix and layers are re-grouped.
    """
    has_dotted_channels = any("." in ch.name for part in parts for ch in part.channels)
    if has_dotted_channels:
        return

    for part in parts:
        part_name = part.header.name
        if not part_name:
            continue
        for ch in part.channels:
            ch.name = f"{part_name}.{ch.name}"
        part.layers = group_channels_into_layers(part.channels)


# --- Tone Mapping Utilities ---


def apply_exposure(rgb: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    """Apply exposure adjustment in stops.

    Args:
        rgb: RGB image data as NumPy array
        exposure: Exposure adjustment in stops

    Returns:
        Exposure-adjusted RGB data
    """
    return rgb * (2.0**exposure)


def apply_gamma(rgb: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Apply gamma correction.

    Args:
        rgb: RGB image data as NumPy array (values should be non-negative)
        gamma: Gamma correction value

    Returns:
        Gamma-corrected RGB data
    """
    return np.power(np.maximum(rgb, 0.0), 1.0 / gamma)


def tone_map_simple(rgb: np.ndarray, exposure: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """Simple tone mapping: exposure + clamp + gamma.

    Args:
        rgb: HDR RGB image data as NumPy array
        exposure: Exposure adjustment in stops
        gamma: Gamma correction value

    Returns:
        Tone-mapped RGB data in [0, 1] range
    """
    rgb = apply_exposure(rgb, exposure)
    rgb = np.clip(rgb, 0.0, 1.0)
    return apply_gamma(rgb, gamma)


def tone_map_reinhard(rgb: np.ndarray, exposure: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """Reinhard tone mapping operator.

    Args:
        rgb: HDR RGB image data as NumPy array
        exposure: Exposure adjustment in stops
        gamma: Gamma correction value

    Returns:
        Tone-mapped RGB data in [0, 1] range
    """
    rgb = apply_exposure(rgb, exposure)
    rgb = rgb / (1.0 + rgb)
    return apply_gamma(rgb, gamma)


def tone_map(
    rgb: np.ndarray,
    method: Literal["simple", "reinhard"] = "simple",
    exposure: float = 0.0,
    gamma: float = 2.2,
) -> np.ndarray:
    """Apply tone mapping using the selected algorithm.

    Args:
        rgb: HDR RGB image data as NumPy array
        method: Tone mapping algorithm
        exposure: Exposure adjustment in stops
        gamma: Gamma correction value

    Returns:
        Tone-mapped RGB data in [0, 1] range

    Raises:
        ValueError: If tone mapping method is not recognized
    """
    match method.lower():
        case "simple":
            return tone_map_simple(rgb, exposure, gamma)
        case "reinhard":
            return tone_map_reinhard(rgb, exposure, gamma)
        case _:
            msg = f"Unknown tone mapping method: '{method}'. Valid options: 'simple', 'reinhard'"
            raise ValueError(msg)


def _ensure_2d(pixels: np.ndarray) -> np.ndarray:
    """Ensure channel pixel data is 2D (H, W), squeezing trailing size-1 dimensions.

    OpenEXR's Python bindings (3.4.x) sometimes return per-channel pixel arrays
    with shape (H, W, 1) instead of the expected (H, W). When three such channels
    are stacked via np.stack([r, g, b], axis=-1), the result is (H, W, 1, 3) — a
    4D array that PIL.Image.fromarray rejects ("Too many dimensions: 4 > 3").

    This helper strips trailing size-1 dimensions so that stacking always produces
    a clean (H, W, 3) RGB array.

    Raises:
        ValueError: If a trailing dimension has size > 1, which would indicate
            genuinely multi-valued data (not a benign broadcasting artifact).
    """
    # Peel off trailing dimensions one at a time, but only if they are size 1.
    # A size-1 trailing dim is just a broadcasting artifact from OpenEXR; a size > 1
    # trailing dim would mean real data that we must not silently discard.
    while pixels.ndim > 2:  # noqa: PLR2004
        trailing_size = pixels.shape[-1]
        if trailing_size != 1:
            msg = (
                f"Cannot squeeze trailing dimension of size {trailing_size} "
                f"(shape {pixels.shape}). Expected size 1 — this may indicate "
                f"unexpected multi-sample or interleaved channel data."
            )
            raise ValueError(msg)
        pixels = pixels.squeeze(axis=-1)
    return pixels


# Canonical channel role mapping, inspired by Nuke's ExrChannelNameToNuke.cpp.
# Maps all known name variants (case-insensitive) to a semantic role so that
# channel assembly is order-independent — a BGR file still produces correct RGB.
_CHANNEL_ROLE_MAP: dict[str, str] = {
    "r": "red",
    "red": "red",
    "g": "green",
    "green": "green",
    "b": "blue",
    "blue": "blue",
    "a": "alpha",
    "alpha": "alpha",
    "y": "luminance",
    "luminance": "luminance",
    "z": "depth",
}


def _normalize_channel_role(name: str) -> str | None:
    """Map a channel name to a canonical role, or None if unrecognized.

    Handles case variations: R/r/Red/RED all map to "red", etc.
    """
    return _CHANNEL_ROLE_MAP.get(name.lower())


def _positional_stack_rgb(pixel_arrays: list[np.ndarray]) -> np.ndarray:
    """Stack 2D pixel arrays positionally into (H, W, 3) RGB.

    Takes the first 3 channels in file order. For fewer than 3 channels:
    2 channels → [ch0, ch1, zeros], 1 channel → grayscale [ch, ch, ch].
    """
    if len(pixel_arrays) >= 3:  # noqa: PLR2004
        return np.stack(pixel_arrays[:3], axis=-1)
    if len(pixel_arrays) == 2:  # noqa: PLR2004
        zeros = np.zeros_like(pixel_arrays[0])
        return np.stack([pixel_arrays[0], pixel_arrays[1], zeros], axis=-1)
    return np.stack([pixel_arrays[0], pixel_arrays[0], pixel_arrays[0]], axis=-1)


def _channels_to_rgb(channels: list[EXRChannel], *, strip_layer_prefix: bool = False) -> np.ndarray:
    """Assemble a (H, W, 3) RGB array from a list of EXR channels.

    Strategy (in priority order):
    1. If any channel's pixels are already 3D with last dim >= 3, treat it as
       interleaved data (e.g. OpenEXR's combined "RGB" channel) and return directly.
    2. Normalize all channels to 2D via _ensure_2d.
    3. Try to find channels by semantic role (red/green/blue) using canonical name
       normalization — this handles R/G/B, r/g/b, Red/Green/Blue, RED/GREEN/BLUE,
       and crucially, BGR or any other ordering.
    4. If role-based assembly fails (fewer than 3 recognized color channels), fall
       back to positional: take channels in file order.
       - 3+ channels: stack first 3
       - 2 channels: stack as [ch0, ch1, zeros]
       - 1 channel: grayscale [ch, ch, ch]

    Args:
        channels: List of EXRChannel objects
        strip_layer_prefix: If True, strip layer prefix before role lookup
            (needed for layer channels named like "beauty.R")

    Returns:
        RGB data as NumPy array (H, W, 3)

    Raises:
        ValueError: If channels list is empty
    """
    if not channels:
        msg = "No channels provided"
        raise ValueError(msg)

    # Step 1: Check for already-interleaved channel data (shape H, W, 3+).
    for ch in channels:
        if ch.pixels.ndim >= 3 and ch.pixels.shape[-1] >= 3:  # noqa: PLR2004
            # Already interleaved — take first 3 bands.
            return ch.pixels[..., :3]

    # Step 2: Normalize all channels to 2D (handles OpenEXR's (H,W,1) quirk).
    normalized: list[tuple[str, np.ndarray]] = []
    for ch in channels:
        short_name = parse_channel_name(ch.name).channel_name if strip_layer_prefix else ch.name
        normalized.append((short_name, _ensure_2d(ch.pixels)))

    # Step 3: Try role-based assembly (red, green, blue).
    by_role: dict[str, np.ndarray] = {}
    for name, pixels in normalized:
        role = _normalize_channel_role(name)
        if role and role not in by_role:
            by_role[role] = pixels

    if all(role in by_role for role in ("red", "green", "blue")):
        return np.stack([by_role["red"], by_role["green"], by_role["blue"]], axis=-1)

    # Step 3b: Luminance-only — Nuke maps Y/y to all three RGB channels (grayscale).
    if "luminance" in by_role:
        lum = by_role["luminance"]
        return np.stack([lum, lum, lum], axis=-1)

    # Step 4: Positional fallback for non-standard channel names (H/S/V, Y/Pb/Pr, etc.).
    return _positional_stack_rgb([pixels for _, pixels in normalized])


def extract_rgb_from_exr_part(part: EXRPart) -> np.ndarray:
    """Extract RGB data from an EXR part.

    Args:
        part: EXRPart containing channel data

    Returns:
        RGB data as NumPy array (H, W, 3)

    Raises:
        ValueError: If no channels are available
    """
    if not part.channels:
        msg = "EXR part has no channels"
        raise ValueError(msg)
    return _channels_to_rgb(part.channels)


def extract_rgb_from_layer(layer: EXRLayer) -> np.ndarray:
    """Extract RGB data from an EXR layer.

    Args:
        layer: EXRLayer containing channel data

    Returns:
        RGB data as NumPy array (H, W, 3)

    Raises:
        ValueError: If layer has no channels
    """
    if not layer.channels:
        msg = f"Layer '{layer.name}' has no channels"
        raise ValueError(msg)
    return _channels_to_rgb(layer.channels, strip_layer_prefix=True)


def generate_exr_preview(  # noqa: PLR0913
    exr_data: EXRData,
    max_width: int = 1024,
    max_height: int = 1024,
    tone_mapping_method: Literal["simple", "reinhard"] = "simple",
    exposure: float = 0.0,
    gamma: float = 2.2,
    part_index: int = 0,
) -> Image.Image:
    """Generate a tone-mapped preview image from EXR data.

    Args:
        exr_data: EXRData containing the loaded EXR file
        max_width: Maximum width for the preview
        max_height: Maximum height for the preview
        tone_mapping_method: Tone mapping algorithm
        exposure: Exposure adjustment in stops
        gamma: Gamma correction value
        part_index: Which part to preview

    Returns:
        PIL Image with tone-mapped sRGB preview

    Raises:
        ValueError: If part_index is out of range
    """
    if part_index >= len(exr_data.parts):
        msg = f"Part index {part_index} out of range (EXR has {len(exr_data.parts)} parts)"
        raise ValueError(msg)

    part = exr_data.parts[part_index]
    rgb = extract_rgb_from_exr_part(part)
    rgb_ldr = tone_map(rgb, tone_mapping_method, exposure, gamma)
    rgb_8bit = np.clip(rgb_ldr * 255.0, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(rgb_8bit, mode="RGB")
    pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return pil_image


def generate_layer_preview(  # noqa: PLR0913
    layer: EXRLayer,
    max_width: int = 512,
    max_height: int = 512,
    tone_mapping_method: Literal["simple", "reinhard"] = "simple",
    exposure: float = 0.0,
    gamma: float = 2.2,
) -> Image.Image:
    """Generate a tone-mapped preview for a single EXR layer.

    Args:
        layer: EXRLayer to preview
        max_width: Maximum width for the preview
        max_height: Maximum height for the preview
        tone_mapping_method: Tone mapping algorithm
        exposure: Exposure adjustment in stops
        gamma: Gamma correction value

    Returns:
        PIL Image with tone-mapped sRGB preview of the layer
    """
    rgb = extract_rgb_from_layer(layer)
    rgb_ldr = tone_map(rgb, tone_mapping_method, exposure, gamma)
    rgb_8bit = np.clip(rgb_ldr * 255.0, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(rgb_8bit, mode="RGB")
    pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return pil_image
