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


class PixelType(StrEnum):
    """OpenEXR pixel types per channel."""

    HALF = "half"
    FLOAT = "float"
    UINT = "uint"


# Mapping dicts from opaque pybind11 OpenEXR enums to our serializable StrEnums.
# OpenEXR enums are C++ wrappers (not str, not int, not JSON-serializable) —
# these dicts are the bridge at the loading boundary.
_COMPRESSION_MAP: dict = {getattr(OpenEXR.Compression, ct.name): ct for ct in CompressionType}
_LINE_ORDER_MAP: dict = {getattr(OpenEXR.LineOrder, lo.name): lo for lo in LineOrderType}
_STORAGE_MAP: dict = {
    OpenEXR.Storage.scanlineimage: StorageType.SCANLINE_IMAGE,
    OpenEXR.Storage.tiledimage: StorageType.TILED_IMAGE,
}
_PIXEL_TYPE_MAP: dict = {
    OpenEXR.PixelType.HALF: PixelType.HALF,
    OpenEXR.PixelType.FLOAT: PixelType.FLOAT,
    OpenEXR.PixelType.UINT: PixelType.UINT,
}


def _map_exr_enum(mapping: dict, exr_value: Any, label: str) -> Any:
    """Look up an OpenEXR enum in a mapping dict with a clear error on failure.

    Raises:
        ValueError: If the OpenEXR enum value is not in the mapping.
    """
    result = mapping.get(exr_value)
    if result is not None:
        return result
    valid = ", ".join(str(v) for v in mapping.values())
    msg = f"Unsupported {label}: {exr_value!r}. Supported values: {valid}"
    raise ValueError(msg)


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
        pixel_type: Original pixel type from EXR file
        x_sampling: Horizontal sampling rate (usually 1)
        y_sampling: Vertical sampling rate (usually 1)
    """

    name: str
    pixels: np.ndarray
    pixel_type: PixelType
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


def _sanitize_name_part(part: str) -> str:
    """Sanitize a name part to match Nuke's ExrChannelNameToNuke behavior.

    Strips leading digits and replaces non-alphanumeric characters with underscores.
    """
    # Strip leading digits
    i = 0
    while i < len(part) and part[i].isdigit():
        i += 1
    part = part[i:]

    # Replace non-alphanumeric with underscores
    return "".join(c if c.isalnum() else "_" for c in part)


def parse_channel_name(full_name: str) -> ChannelNameParts:
    """Parse EXR channel name into layer and channel components.

    Matches Nuke's ExrChannelNameToNuke.cpp behavior:
    - Splits on '.' with max 3 parts (2 splits)
    - Strips leading digits from each part
    - Replaces non-alphanumeric characters with underscores
    - All parts except last form the layer name (joined with '_')
    - Last part is the channel name
    - "Ci" layer (RenderMan default) maps to default layer

    Args:
        full_name: Full channel name (e.g., "beauty.R", "R", "View Layer.AO.R")

    Returns:
        ChannelNameParts with layer_name and channel_name

    Examples:
        "R" → ChannelNameParts("", "R")
        "beauty.R" → ChannelNameParts("beauty", "R")
        "View Layer.AO.R" → ChannelNameParts("View_Layer_AO", "R")
        "diffuse.indirect.R" → ChannelNameParts("diffuse_indirect", "R")
        "Ci.R" → ChannelNameParts("", "R")
    """
    # Split on '.', max 3 parts (matching Nuke's split that stops after 2 separators)
    parts = full_name.split(".", maxsplit=2)

    # Sanitize each part and drop empty results
    sanitized = [s for p in parts if (s := _sanitize_name_part(p))]

    if len(sanitized) <= 1:
        channel_name = sanitized[0] if sanitized else "unnamed"
        return ChannelNameParts(layer_name="", channel_name=channel_name)

    channel_name = sanitized[-1]
    layer_name = "_".join(sanitized[:-1])

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
    layers.sort(key=lambda layer: (layer.name != "", layer.name))

    return layers


def attempt_read_exr(file_path: str) -> EXRData:
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
        with OpenEXR.File(file_path, separate_channels=True) as exr:
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

                # Build EXRChannel objects using structured Channel attributes.
                # raw_channels values are Channel objects with .type(), .xSampling, .ySampling.
                channels_list = []
                for ch_name, ch_obj in raw_channels.items():
                    pixel_type = _map_exr_enum(_PIXEL_TYPE_MAP, ch_obj.type(), "pixel type")

                    exr_channel = EXRChannel(
                        name=ch_name,
                        pixels=ch_obj.pixels.copy(),  # Copy before file closes
                        pixel_type=pixel_type,
                        x_sampling=ch_obj.xSampling,
                        y_sampling=ch_obj.ySampling,
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

                # Normalize windows so display origin is at (0, 0)
                data_window, display_window = _normalize_windows(data_window, display_window)

                # Extract screen window center (numpy array -> tuple)
                screen_center = raw_header["screenWindowCenter"]
                screen_center_tuple = (float(screen_center[0]), float(screen_center[1]))

                # Map OpenEXR enums to our StrEnums
                compression = _map_exr_enum(_COMPRESSION_MAP, raw_header["compression"], "compression")
                line_order = _map_exr_enum(_LINE_ORDER_MAP, raw_header["lineOrder"], "line order")
                storage_type = _map_exr_enum(_STORAGE_MAP, raw_header["type"], "storage type")

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
                custom_attrs = {
                    k: _convert_attribute_value(v) for k, v in raw_header.items() if k not in required_attrs
                }

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


def _convert_attribute_value(value: Any) -> Any:
    """Convert an OpenEXR attribute value to a clean Python type.

    Handles common EXR metadata types so they serialize to JSON cleanly
    instead of producing opaque repr() strings.
    """
    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (tuple, list)):
        return [_convert_attribute_value(v) for v in value]

    # NumPy scalar or array
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()

    # Fraction-like (framesPerSecond is typically a Fraction)
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return {"numerator": value.numerator, "denominator": value.denominator}

    return str(value)


def _normalize_windows(
    data_window: WindowCoordinates,
    display_window: WindowCoordinates,
) -> tuple[WindowCoordinates, WindowCoordinates]:
    """Normalize windows so display window origin is at (0, 0).

    Matches Nuke's offset_negative_display_window behavior (exrReader.cpp:1755-1793).
    When the display window doesn't start at (0, 0), both windows are shifted so that
    the display window minimum becomes the origin. This ensures coordinates are
    consistent for downstream operations.

    Returns:
        Tuple of (normalized_data_window, normalized_display_window).
    """
    x_offset = display_window.xmin
    y_offset = display_window.ymin

    if x_offset == 0 and y_offset == 0:
        return data_window, display_window

    normalized_display = WindowCoordinates(
        xmin=0,
        ymin=0,
        xmax=display_window.xmax - x_offset,
        ymax=display_window.ymax - y_offset,
    )
    normalized_data = WindowCoordinates(
        xmin=data_window.xmin - x_offset,
        ymin=data_window.ymin - y_offset,
        xmax=data_window.xmax - x_offset,
        ymax=data_window.ymax - y_offset,
    )
    return normalized_data, normalized_display


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
    # Nuke also checks for FULL_LAYER_NAMES metadata flag — if set, channels already
    # contain full layer paths and part names should not be prepended.
    has_full_layer_names = any(part.header.custom.get("fullLayerNames") for part in parts)
    if has_full_layer_names:
        return

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


def tone_map_filmic(rgb: np.ndarray, exposure: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """Filmic tone mapping based on John Hable's Uncharted 2 curve.

    Args:
        rgb: HDR RGB image data as NumPy array
        exposure: Exposure adjustment in stops
        gamma: Gamma correction value

    Returns:
        Tone-mapped RGB data in [0, 1] range
    """
    rgb = apply_exposure(rgb, exposure)

    # Hable filmic curve parameters
    a = 0.22  # Shoulder strength
    b = 0.30  # Linear strength
    c = 0.10  # Linear angle
    d = 0.20  # Toe strength
    e = 0.01  # Toe numerator
    f = 0.30  # Toe denominator

    rgb_mapped = ((rgb * (a * rgb + c * b) + d * e) / (rgb * (a * rgb + b) + d * f)) - e / f

    # Normalize against white point
    white_point = 11.2
    white_mapped = (
        (white_point * (a * white_point + c * b) + d * e) / (white_point * (a * white_point + b) + d * f)
    ) - e / f
    rgb_mapped = rgb_mapped / white_mapped

    return apply_gamma(rgb_mapped, gamma)


def tone_map(
    rgb: np.ndarray,
    method: Literal["simple", "reinhard", "filmic"] = "simple",
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
        case "filmic":
            return tone_map_filmic(rgb, exposure, gamma)
        case _:
            msg = f"Unknown tone mapping method: '{method}'. Valid options: 'simple', 'reinhard', 'filmic'"
            raise ValueError(msg)


def _ensure_2d(pixels: np.ndarray) -> np.ndarray:
    """Validate that channel pixel data is 2D (H, W).

    With separate_channels=True, OpenEXR should always return (H, W) arrays.
    Any other shape indicates unexpected data that should not be silently coerced.

    Raises:
        ValueError: If pixel data is not 2D.
    """
    if pixels.ndim != 2:  # noqa: PLR2004
        msg = (
            f"Expected 2D channel data (H, W), got shape {pixels.shape}. "
            f"This may indicate interleaved or multi-sample data."
        )
        raise ValueError(msg)
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

    Expects separate per-channel data (loaded with separate_channels=True).

    Strategy:
    1. Validate all channels are 2D (H, W) via _ensure_2d.
    2. Try role-based assembly (red/green/blue) using canonical name normalization —
       handles R/G/B, r/g/b, Red/Green/Blue, RED/GREEN/BLUE, and BGR ordering.
       Also handles Y/luminance as grayscale.
    3. Positional fallback for non-standard channel names (H/S/V, Y/Pb/Pr, etc.):
       - 3+ channels: stack first 3
       - 2 channels: stack as [ch0, ch1, zeros]
       - 1 channel: grayscale [ch, ch, ch]

    Args:
        channels: List of EXRChannel objects with 2D pixel data
        strip_layer_prefix: If True, strip layer prefix before role lookup
            (needed for layer channels named like "beauty.R")

    Returns:
        RGB data as NumPy array (H, W, 3)

    Raises:
        ValueError: If channels list is empty or any channel is not 2D
    """
    if not channels:
        msg = "No channels provided"
        raise ValueError(msg)

    # Step 1: Validate all channels are 2D.
    normalized: list[tuple[str, np.ndarray]] = []
    for ch in channels:
        short_name = parse_channel_name(ch.name).channel_name if strip_layer_prefix else ch.name
        normalized.append((short_name, _ensure_2d(ch.pixels)))

    # Step 2: Try role-based assembly (red, green, blue).
    by_role: dict[str, np.ndarray] = {}
    for name, pixels in normalized:
        role = _normalize_channel_role(name)
        if role and role not in by_role:
            by_role[role] = pixels

    if all(role in by_role for role in ("red", "green", "blue")):
        return np.stack([by_role["red"], by_role["green"], by_role["blue"]], axis=-1)

    # Step 2b: Luminance-only — Nuke maps Y/y to all three RGB channels (grayscale).
    if "luminance" in by_role:
        lum = by_role["luminance"]
        return np.stack([lum, lum, lum], axis=-1)

    # Step 3: Positional fallback for non-standard channel names (H/S/V, Y/Pb/Pr, etc.).
    return _positional_stack_rgb([pixels for _, pixels in normalized])


def extract_rgb_from_exr_part(part: EXRPart) -> np.ndarray:
    """Extract RGB data from an EXR part's top-level rgba channels.

    Matches Nuke's behavior: finds bare R, G, B channels (no layer prefix)
    directly in the part's channel list. Falls back to the first layer if
    no top-level rgba channels exist.

    Args:
        part: EXRPart containing channel and layer data

    Returns:
        RGB data as NumPy array (H, W, 3)

    Raises:
        ValueError: If no channels are available
    """
    if not part.channels:
        msg = "EXR part has no channels"
        raise ValueError(msg)

    # Find top-level rgba channels (no layer prefix) by semantic role
    rgba_channels: list[EXRChannel] = []
    for ch in part.channels:
        parsed = parse_channel_name(ch.name)
        if parsed.layer_name != "":
            continue
        role = _normalize_channel_role(parsed.channel_name)
        if role in ("red", "green", "blue", "alpha"):
            rgba_channels.append(ch)

    if rgba_channels:
        return _channels_to_rgb(rgba_channels, strip_layer_prefix=True)

    # No top-level rgba — fall back to first layer
    if not part.layers:
        msg = "EXR part has no layers"
        raise ValueError(msg)
    return extract_rgb_from_layer(part.layers[0])


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
    tone_mapping_method: Literal["simple", "reinhard", "filmic"] = "simple",
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
    tone_mapping_method: Literal["simple", "reinhard", "filmic"] = "simple",
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
