"""EXR data structures, I/O operations, and tone mapping utilities.

Uses OpenImageIO (OIIO) for two-phase lazy loading:
- scan_exr(): reads headers only (no pixels) — fast, for UI population
- load_channels(): reads specific channels on demand — selective, for pixel access
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

import numpy as np
import OpenImageIO as oiio  # type: ignore[import-not-found]
from PIL import Image


# --- Enums ---


class CompressionType(StrEnum):
    """OpenEXR compression types."""

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
    """OpenEXR line order types."""

    INCREASING_Y = "INCREASING_Y"
    DECREASING_Y = "DECREASING_Y"
    RANDOM_Y = "RANDOM_Y"


class StorageType(StrEnum):
    """OpenEXR storage types."""

    SCANLINE_IMAGE = "scanlineimage"
    TILED_IMAGE = "tiledimage"


class PixelType(StrEnum):
    """OpenEXR pixel types per channel."""

    HALF = "half"
    FLOAT = "float"
    UINT = "uint"


class ToneMappingMethod(StrEnum):
    """Tone mapping algorithms for HDR to LDR conversion."""

    SIMPLE = "simple"
    REINHARD = "reinhard"
    FILMIC = "filmic"


# --- OIIO string → StrEnum mappings ---

_OIIO_COMPRESSION_MAP: dict[str, CompressionType] = {
    "none": CompressionType.NO_COMPRESSION,
    "rle": CompressionType.RLE_COMPRESSION,
    "zips": CompressionType.ZIPS_COMPRESSION,
    "zip": CompressionType.ZIP_COMPRESSION,
    "piz": CompressionType.PIZ_COMPRESSION,
    "pxr24": CompressionType.PXR24_COMPRESSION,
    "b44": CompressionType.B44_COMPRESSION,
    "b44a": CompressionType.B44A_COMPRESSION,
    "dwaa": CompressionType.DWAA_COMPRESSION,
    "dwab": CompressionType.DWAB_COMPRESSION,
}

_OIIO_PIXEL_TYPE_MAP: dict[str, PixelType] = {
    "half": PixelType.HALF,
    "float": PixelType.FLOAT,
    "uint32": PixelType.UINT,
    "uint16": PixelType.UINT,
    "uint8": PixelType.UINT,
}

_OIIO_LINE_ORDER_MAP: dict[str, LineOrderType] = {
    "increasingY": LineOrderType.INCREASING_Y,
    "decreasingY": LineOrderType.DECREASING_Y,
    "randomY": LineOrderType.RANDOM_Y,
}


# --- OIIO attribute name constants ---
# Used in both _HEADER_SKIP_ATTRS and _build_header_from_spec to avoid
# silent breakage from typos in magic strings.

_ATTR_COMPRESSION = "compression"
_ATTR_LINE_ORDER = "openexr:lineOrder"
_ATTR_CHUNK_COUNT = "openexr:chunkCount"
_ATTR_NAME = "name"
_ATTR_PIXEL_ASPECT_RATIO = "PixelAspectRatio"
_ATTR_SCREEN_WINDOW_CENTER = "screenWindowCenter"
_ATTR_SCREEN_WINDOW_WIDTH = "screenWindowWidth"


# OIIO spec attributes that already map to dedicated EXRHeader fields.
# Excluded from EXRHeader.custom to avoid duplication.
_HEADER_SKIP_ATTRS: set[str] = {
    _ATTR_COMPRESSION,
    _ATTR_LINE_ORDER,
    _ATTR_CHUNK_COUNT,
    _ATTR_NAME,
    "oiio:subimagename",
    "oiio:subimages",
    _ATTR_PIXEL_ASPECT_RATIO,
    _ATTR_SCREEN_WINDOW_CENTER,
    _ATTR_SCREEN_WINDOW_WIDTH,
}


def _map_oiio_string(mapping: dict[str, Any], oiio_value: str, label: str) -> Any:
    """Look up an OIIO string in a mapping dict with a clear error on failure.

    Args:
        mapping: Dict mapping OIIO string values to our StrEnum members
        oiio_value: The string value returned by OIIO
        label: Human-readable name for error messages (e.g., "compression", "pixel type")

    Returns:
        The matched StrEnum member

    Raises:
        ValueError: If oiio_value is not found in the mapping
    """
    result = mapping.get(oiio_value)
    if result is not None:
        return result
    valid = ", ".join(f"'{k}'" for k in mapping)
    msg = f"Unsupported {label}: '{oiio_value}'. Supported values: {valid}"
    raise ValueError(msg)


# --- Data Structures ---


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


class NormalizedWindows(NamedTuple):
    """Result of normalizing EXR data/display windows.

    Attributes:
        data: Data window with offset applied
        display: Display window with origin at (0, 0)
    """

    data: WindowCoordinates
    display: WindowCoordinates


class _NormalizedChannel(NamedTuple):
    """Channel name paired with its 2D pixel data for RGB assembly."""

    name: str
    pixels: np.ndarray


@dataclass
class EXRChannelInfo:
    """Channel metadata from an OpenEXR file (no pixel data).

    Attributes:
        name: Channel name (e.g., "R", "G", "B", "Z", "beauty.R")
        pixel_type: Original pixel type from EXR file
        channel_index: Index within the part's channel list (for OIIO reads)
        x_sampling: Horizontal sampling rate
        y_sampling: Vertical sampling rate
    """

    name: str
    pixel_type: PixelType
    channel_index: int
    x_sampling: int
    y_sampling: int


@dataclass
class EXRChannelPixelData:
    """Channel metadata paired with loaded pixel data.

    Attributes:
        info: Channel metadata
        pixels: Pixel data as NumPy array (float32)
    """

    info: EXRChannelInfo
    pixels: np.ndarray


@dataclass
class EXRLayer:
    """Layer grouping channels with same prefix.

    Attributes:
        name: Layer name (empty string for default/main layer)
        channels: List of channels in this layer
    """

    name: str
    channels: list[EXRChannelInfo]


@dataclass
class EXRHeader:
    """Header metadata from an OpenEXR file part.

    Attributes:
        compression: Compression algorithm used for this part
        line_order: Scanline storage order
        data_window: Bounding box of actual pixel data
        display_window: Intended display area (may differ from data window)
        pixel_aspect_ratio: Width/height ratio of a single pixel (1.0 = square)
        screen_window_center: Center of the screen window in NDC
        screen_window_width: Width of the screen window in NDC
        storage_type: Whether the part is scanline or tiled
        name: Part name (empty string for single-part files)
        chunk_count: Number of chunks (present in multi-part files, None otherwise)
        custom: Non-standard header attributes (key-value pairs)
    """

    compression: CompressionType
    line_order: LineOrderType
    data_window: WindowCoordinates
    display_window: WindowCoordinates
    pixel_aspect_ratio: float
    screen_window_center: tuple[float, float]
    screen_window_width: float
    storage_type: StorageType
    name: str
    chunk_count: int | None
    custom: dict[str, Any]


@dataclass
class EXRPart:
    """Single part from an OpenEXR file.

    Attributes:
        channels: All channels in this part (pixels may or may not be loaded)
        layers: Channels grouped by layer name prefix
        header: Full EXR header metadata for this part
        index: Zero-based part index within the file
        width: Image width in pixels (derived from data window)
        height: Image height in pixels (derived from data window)
    """

    channels: list[EXRChannelInfo]
    layers: list[EXRLayer]
    header: EXRHeader
    index: int
    width: int
    height: int


@dataclass
class EXRData:
    """All data extracted from an OpenEXR file.

    Returned by scan_exr(). After scanning, all channels have pixels=None.
    Use load_layer_pixels() or load_channels() to populate pixel data on demand.

    Attributes:
        parts: List of parts in the file (single-part files have one entry)
    """

    parts: list[EXRPart]


# --- Channel Name Parsing (Nuke-compatible) ---


def parse_channel_name(full_name: str) -> ChannelNameParts:
    """Parse EXR channel name into layer and channel components.

    Matches Nuke's ExrChannelNameToNuke.cpp behavior:
    - Splits on '.' with max 3 parts (2 splits)
    - Strips leading digits from each part
    - Replaces non-alphanumeric characters with underscores
    - All parts except last form the layer name (joined with '_')
    - Last part is the channel name
    - "Ci" layer (RenderMan default) maps to default layer

    Examples:
        "R" → ChannelNameParts("", "R")
        "beauty.R" → ChannelNameParts("beauty", "R")
        "View Layer.AO.R" → ChannelNameParts("View_Layer_AO", "R")
        "Ci.R" → ChannelNameParts("", "R")
    """
    parts = full_name.split(".", maxsplit=2)
    sanitized: list[str] = []
    for p in parts:
        s = _sanitize_name_part(p)
        if s:
            sanitized.append(s)

    if len(sanitized) <= 1:
        channel_name = sanitized[0] if sanitized else "unnamed"
        return ChannelNameParts(layer_name="", channel_name=channel_name)

    channel_name = sanitized[-1]
    layer_name = "_".join(sanitized[:-1])

    if layer_name == "Ci":
        layer_name = ""

    return ChannelNameParts(layer_name=layer_name, channel_name=channel_name)


def group_channels_into_layers(channels: list[EXRChannelInfo]) -> list[EXRLayer]:
    """Group channels by layer name.

    Returns list of EXRLayer objects, sorted by layer name (default layer first).
    """
    layers_dict: dict[str, list[EXRChannelInfo]] = {}

    for channel in channels:
        parsed = parse_channel_name(channel.name)
        layer_name = parsed.layer_name

        if layer_name not in layers_dict:
            layers_dict[layer_name] = []
        layers_dict[layer_name].append(channel)

    layers: list[EXRLayer] = []
    for layer_name, channels_list in layers_dict.items():
        layers.append(EXRLayer(name=layer_name, channels=channels_list))
    layers.sort(key=lambda layer: (layer.name != "", layer.name))
    return layers


def _sanitize_name_part(part: str) -> str:
    """Sanitize a name part to match Nuke's ExrChannelNameToNuke behavior.

    Strips leading digits and replaces non-alphanumeric characters with underscores.
    """
    i = 0
    while i < len(part) and part[i].isdigit():
        i += 1
    part = part[i:]
    return "".join(c if c.isalnum() else "_" for c in part)


# --- OIIO-based I/O ---


def scan_exr(file_path: str) -> EXRData:
    """Scan an EXR file's headers without loading pixel data.

    Uses OIIO to read metadata for all parts. Returns EXRData with full
    structure (headers, channels, layers) but pixels=None on all channels.

    Args:
        file_path: Path to the EXR file

    Returns:
        EXRData with metadata only (no pixel data)

    Raises:
        ValueError: If file_path is empty or file has no valid parts
        RuntimeError: If OIIO fails to open the file
    """
    if not file_path:
        msg = "file_path must not be empty"
        raise ValueError(msg)

    inp = oiio.ImageInput.open(file_path)
    if not inp:
        msg = f"Failed to open EXR file '{file_path}': {oiio.geterror()}"
        raise RuntimeError(msg)

    try:
        exr_parts: list[EXRPart] = []
        subimage_idx = 0

        while inp.seek_subimage(subimage_idx, 0):
            spec = inp.spec()

            # Build channel list with metadata (no pixels)
            channels_list: list[EXRChannelInfo] = []
            for ch_idx in range(spec.nchannels):
                ch_name = spec.channelnames[ch_idx]
                ch_format = str(spec.channelformat(ch_idx))
                pixel_type = _map_oiio_string(_OIIO_PIXEL_TYPE_MAP, ch_format, "pixel type")

                channels_list.append(
                    EXRChannelInfo(
                        name=ch_name,
                        pixel_type=pixel_type,
                        channel_index=ch_idx,
                        x_sampling=1,
                        y_sampling=1,
                    )
                )

            if not channels_list:
                msg = f"EXR file part {subimage_idx} has no channels: {file_path}"
                raise ValueError(msg)

            # Build header from OIIO spec
            header = _build_header_from_spec(spec)

            # Build EXRPart
            data_win = header.data_window
            width = data_win.xmax - data_win.xmin + 1
            height = data_win.ymax - data_win.ymin + 1

            exr_part = EXRPart(
                channels=channels_list,
                layers=group_channels_into_layers(channels_list),
                header=header,
                index=subimage_idx,
                width=width,
                height=height,
            )
            exr_parts.append(exr_part)
            subimage_idx += 1

        if not exr_parts:
            msg = f"EXR file has no parts: {file_path}"
            raise ValueError(msg)

        if len(exr_parts) > 1:
            _apply_legacy_part_name_prefix(exr_parts)

        return EXRData(parts=exr_parts)

    finally:
        inp.close()


def _group_contiguous_runs(indices: list[int]) -> list[list[int]]:
    """Group sorted indices into contiguous runs.

    For example, [0, 1, 2, 5, 6, 9] becomes [[0, 1, 2], [5, 6], [9]].
    Used to batch OIIO channel reads for efficiency.
    """
    sorted_indices = sorted(indices)
    runs: list[list[int]] = []
    current_run: list[int] = []

    for idx in sorted_indices:
        if current_run and idx != current_run[-1] + 1:
            runs.append(current_run)
            current_run = []
        current_run.append(idx)

    if current_run:
        runs.append(current_run)

    return runs


def load_channels(
    file_path: str,
    part_index: int,
    channel_indices: list[int],
) -> dict[int, np.ndarray]:
    """Load specific channels from an EXR file by index.

    Opens the file, seeks to the part, and reads only the requested
    channel indices. Groups contiguous indices into single read calls
    for efficiency.

    Args:
        file_path: Path to the EXR file
        part_index: Part index (maps to OIIO subimage)
        channel_indices: List of channel indices to load

    Returns:
        Dict mapping channel index → 2D float32 array (H, W)

    Raises:
        RuntimeError: If OIIO fails to open or read the file
        ValueError: If channel_indices is empty
    """
    if not channel_indices:
        msg = "channel_indices must not be empty"
        raise ValueError(msg)

    inp = oiio.ImageInput.open(file_path)
    if not inp:
        msg = f"Failed to open EXR file '{file_path}': {oiio.geterror()}"
        raise RuntimeError(msg)

    try:
        if not inp.seek_subimage(part_index, 0):
            msg = f"Failed to seek to part {part_index} in '{file_path}'"
            raise RuntimeError(msg)

        spec = inp.spec()
        result: dict[int, np.ndarray] = {}

        # Group indices into contiguous runs for efficient reads
        runs = _group_contiguous_runs(channel_indices)
        for run in runs:
            chbegin = run[0]
            chend = run[-1] + 1

            pixels = inp.read_image(part_index, 0, chbegin, chend, "float")
            if pixels is None:
                msg = f"Failed to read channels {chbegin}-{chend} from part {part_index}: {oiio.geterror()}"
                raise RuntimeError(msg)

            # Split multi-channel result into individual 2D arrays
            if pixels.ndim == 3:  # noqa: PLR2004
                for i, ch_idx in enumerate(run):
                    result[ch_idx] = pixels[:, :, i]
            else:
                # Single channel — already 2D or (H, W, 1)
                result[run[0]] = pixels.reshape(spec.height, spec.width)

        return result

    finally:
        inp.close()


def load_layer_pixels(
    file_path: str,
    part_index: int,
    channels: list[EXRChannelInfo],
) -> list[EXRChannelPixelData]:
    """Load pixel data for a list of channels and return paired results.

    Convenience wrapper around load_channels() that pairs each channel's
    metadata with its loaded pixel data.

    Args:
        file_path: Path to the EXR file
        part_index: Part index (maps to OIIO subimage)
        channels: Channel metadata (channel_index used for OIIO reads)

    Returns:
        List of EXRChannelPixelData pairing each channel's info with its pixels
    """
    indices = [ch.channel_index for ch in channels]
    pixel_data = load_channels(file_path, part_index, indices)

    result: list[EXRChannelPixelData] = []
    for ch in channels:
        result.append(EXRChannelPixelData(info=ch, pixels=pixel_data[ch.channel_index]))  # noqa: PERF401
    return result


def _require_string_attribute(spec: oiio.ImageSpec, name: str) -> str:
    """Get a required string attribute from an OIIO ImageSpec.

    Args:
        spec: OIIO ImageSpec to query
        name: Attribute name to look up

    Returns:
        The attribute's string value

    Raises:
        ValueError: If the attribute is not present in the spec
    """
    sentinel = "__MISSING__"
    value = spec.get_string_attribute(name, sentinel)
    if value == sentinel:
        msg = f"Required EXR header attribute '{name}' is missing"
        raise ValueError(msg)
    return value


def _build_header_from_spec(spec: oiio.ImageSpec) -> EXRHeader:
    """Build an EXRHeader from an OIIO ImageSpec.

    Extracts all standard EXR header fields from the OIIO spec, normalizes
    windows so display origin is at (0, 0), and collects non-standard
    attributes into the custom dict.

    Args:
        spec: OIIO ImageSpec from an open EXR file part

    Returns:
        Fully populated EXRHeader
    """
    # Data window from spec coordinates
    data_window = WindowCoordinates(
        xmin=spec.x,
        ymin=spec.y,
        xmax=spec.x + spec.width - 1,
        ymax=spec.y + spec.height - 1,
    )

    # Display window from full coordinates
    display_window = WindowCoordinates(
        xmin=spec.full_x,
        ymin=spec.full_y,
        xmax=spec.full_x + spec.full_width - 1,
        ymax=spec.full_y + spec.full_height - 1,
    )

    # Normalize windows so display origin is at (0, 0)
    normalized = _normalize_windows(data_window, display_window)
    data_window = normalized.data
    display_window = normalized.display

    # Compression (required by EXR spec)
    comp_str = _require_string_attribute(spec, _ATTR_COMPRESSION)
    compression = _map_oiio_string(_OIIO_COMPRESSION_MAP, comp_str, "compression")

    # Line order (required by EXR spec)
    lo_str = _require_string_attribute(spec, _ATTR_LINE_ORDER)
    line_order = _map_oiio_string(_OIIO_LINE_ORDER_MAP, lo_str, "line order")

    # Storage type — OIIO doesn't expose this directly for reads,
    # but tiled images have tile dimensions > 0
    if spec.tile_width > 0:
        storage_type = StorageType.TILED_IMAGE
    else:
        storage_type = StorageType.SCANLINE_IMAGE

    # Screen window
    screen_center_attr = None
    for attr in spec.extra_attribs:
        if attr.name == _ATTR_SCREEN_WINDOW_CENTER:
            screen_center_attr = attr.value
            break
    if screen_center_attr is not None:
        screen_center = (float(screen_center_attr[0]), float(screen_center_attr[1]))  # type: ignore[index]
    else:
        screen_center = (0.0, 0.0)

    # Optional per EXR spec — defaults are spec-defined
    screen_width = spec.get_float_attribute(_ATTR_SCREEN_WINDOW_WIDTH, 1.0)
    pixel_aspect = spec.get_float_attribute(_ATTR_PIXEL_ASPECT_RATIO, 1.0)

    # Part name and chunk count
    part_name = spec.get_string_attribute(_ATTR_NAME, "")
    chunk_count_val = spec.get_int_attribute(_ATTR_CHUNK_COUNT, -1)
    chunk_count = chunk_count_val if chunk_count_val >= 0 else None

    # Custom attributes — exclude known/required attributes
    custom: dict[str, Any] = {}
    for attr in spec.extra_attribs:
        if attr.name not in _HEADER_SKIP_ATTRS:
            custom[attr.name] = _convert_attribute_value(attr.value)

    return EXRHeader(
        compression=compression,
        line_order=line_order,
        data_window=data_window,
        display_window=display_window,
        pixel_aspect_ratio=pixel_aspect,
        screen_window_center=screen_center,
        screen_window_width=screen_width,
        storage_type=storage_type,
        name=part_name,
        chunk_count=chunk_count,
        custom=custom,
    )


def _convert_attribute_value(value: Any) -> Any:
    """Convert an OIIO attribute value to a serializable Python type.

    OIIO returns attributes as various types (NumPy scalars, byte strings,
    arrays). This normalizes them to plain Python types suitable for JSON
    serialization or display.

    Args:
        value: Raw attribute value from OIIO's extra_attribs

    Returns:
        Python-native equivalent (str, int, float, bool, list, or str fallback)
    """
    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, (tuple, list)):
        converted: list[Any] = []
        for v in value:
            converted.append(_convert_attribute_value(v))  # noqa: PERF401
        return converted

    # NumPy scalar or array
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()

    return str(value)


def _normalize_windows(
    data_window: WindowCoordinates,
    display_window: WindowCoordinates,
) -> NormalizedWindows:
    """Normalize EXR windows so display window origin is at (0, 0).

    EXR files can have non-zero display window origins. This shifts both
    windows by the display window's offset so the display origin lands at
    (0, 0) while preserving the relative position of the data window.
    Matches Nuke's offset_negative_display_window behavior.

    Args:
        data_window: Original data window coordinates from EXR header
        display_window: Original display window coordinates from EXR header

    Returns:
        NormalizedWindows with both windows offset-adjusted
    """
    x_offset = display_window.xmin
    y_offset = display_window.ymin

    if x_offset == 0 and y_offset == 0:
        return NormalizedWindows(data=data_window, display=display_window)

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
    return NormalizedWindows(data=normalized_data, display=normalized_display)


def _apply_legacy_part_name_prefix(parts: list[EXRPart]) -> None:
    """Detect and handle legacy multi-part files that use part names as layer names.

    Nuke auto-detects legacy multi-part files where the part name stores the
    layer name (channels are just R, G, B without '.' separators). If no
    channels in any part contain a '.' separator, the part's header name is
    prepended as a layer prefix and layers are re-grouped.
    """
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
    """Apply exposure adjustment in stops (each stop doubles/halves brightness).

    Args:
        rgb: Float pixel data, any shape
        exposure: Exposure adjustment in stops (0.0 = no change)

    Returns:
        Exposure-adjusted pixel data (same shape as input)
    """
    return rgb * (2.0**exposure)


def apply_gamma(rgb: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Apply gamma correction (linear to display transfer function).

    Negative values are clamped to zero before the power function.

    Args:
        rgb: Float pixel data, any shape
        gamma: Gamma value (2.2 approximates sRGB)

    Returns:
        Gamma-corrected pixel data (same shape as input)
    """
    return np.power(np.maximum(rgb, 0.0), 1.0 / gamma)


def tone_map_simple(rgb: np.ndarray, exposure: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """Simple tone mapping: exposure, clamp [0, 1], gamma.

    Clips HDR values to [0, 1] after exposure. Fast but loses highlight detail.

    Args:
        rgb: HDR float pixel data, any shape
        exposure: Exposure in stops
        gamma: Gamma correction value

    Returns:
        LDR pixel data in [0, 1] range
    """
    rgb = apply_exposure(rgb, exposure)
    rgb = np.clip(rgb, 0.0, 1.0)
    return apply_gamma(rgb, gamma)


def tone_map_reinhard(rgb: np.ndarray, exposure: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """Reinhard tone mapping operator.

    Applies the global Reinhard curve: L / (1 + L). Compresses HDR range
    smoothly — preserves more highlight detail than simple clamp.

    Args:
        rgb: HDR float pixel data, any shape
        exposure: Exposure in stops
        gamma: Gamma correction value

    Returns:
        LDR pixel data in [0, 1] range
    """
    rgb = apply_exposure(rgb, exposure)
    rgb = rgb / (1.0 + rgb)
    return apply_gamma(rgb, gamma)


def tone_map_filmic(rgb: np.ndarray, exposure: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """Filmic tone mapping based on John Hable's Uncharted 2 curve.

    S-shaped curve with shoulder rolloff and toe. Produces a cinematic look
    with controlled highlight compression. Uses a fixed white point of 11.2.

    Args:
        rgb: HDR float pixel data, any shape
        exposure: Exposure in stops
        gamma: Gamma correction value

    Returns:
        LDR pixel data in [0, 1] range
    """
    rgb = apply_exposure(rgb, exposure)

    a = 0.22  # Shoulder strength
    b = 0.30  # Linear strength
    c = 0.10  # Linear angle
    d = 0.20  # Toe strength
    e = 0.01  # Toe numerator
    f = 0.30  # Toe denominator

    rgb_mapped = ((rgb * (a * rgb + c * b) + d * e) / (rgb * (a * rgb + b) + d * f)) - e / f

    white_point = 11.2
    white_mapped = (
        (white_point * (a * white_point + c * b) + d * e) / (white_point * (a * white_point + b) + d * f)
    ) - e / f
    rgb_mapped = rgb_mapped / white_mapped

    return apply_gamma(rgb_mapped, gamma)


def tone_map(
    rgb: np.ndarray,
    method: ToneMappingMethod = ToneMappingMethod.SIMPLE,
    exposure: float = 0.0,
    gamma: float = 2.2,
) -> np.ndarray:
    """Apply tone mapping using the selected algorithm.

    Args:
        rgb: HDR float pixel data, any shape
        method: Tone mapping algorithm — "simple", "reinhard", or "filmic"
        exposure: Exposure in stops
        gamma: Gamma correction value

    Returns:
        LDR pixel data in [0, 1] range

    Raises:
        ValueError: If tone mapping method is not recognized
    """
    match method:
        case ToneMappingMethod.SIMPLE:
            return tone_map_simple(rgb, exposure, gamma)
        case ToneMappingMethod.REINHARD:
            return tone_map_reinhard(rgb, exposure, gamma)
        case ToneMappingMethod.FILMIC:
            return tone_map_filmic(rgb, exposure, gamma)
        case _:
            msg = f"Unknown tone mapping method: '{method}'. Valid: {[m.value for m in ToneMappingMethod]}"
            raise ValueError(msg)


# --- Channel Role Mapping & RGB Assembly ---


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


def normalize_channel_role(name: str) -> str | None:
    """Map a channel name to a canonical role.

    Recognizes standard EXR channel names and their abbreviations:
    R/red, G/green, B/blue, A/alpha, Y/luminance, Z/depth.

    Args:
        name: Channel name (case-insensitive)

    Returns:
        Canonical role string, or None if the name is not a recognized channel role
    """
    return _CHANNEL_ROLE_MAP.get(name.lower())


def extract_rgb_from_channels(channels: list[EXRChannelPixelData]) -> np.ndarray:
    """Extract RGB data from loaded channels, matching Nuke's behavior.

    Looks for bare R, G, B, A channels (no layer prefix) first. If none
    are found, falls back to using all channels positionally.

    Args:
        channels: Loaded channel pixel data

    Returns:
        (H, W, 3) float32 RGB array

    Raises:
        ValueError: If no channels are provided
    """
    if not channels:
        msg = "No channels provided"
        raise ValueError(msg)

    rgba_channels: list[EXRChannelPixelData] = []
    for ch in channels:
        parsed = parse_channel_name(ch.info.name)
        if parsed.layer_name != "":
            continue
        role = normalize_channel_role(parsed.channel_name)
        if role in ("red", "green", "blue", "alpha"):
            rgba_channels.append(ch)

    if rgba_channels:
        return _channels_to_rgb(rgba_channels, strip_layer_prefix=True)

    return _channels_to_rgb(channels, strip_layer_prefix=True)


def extract_rgb_from_layer(channels: list[EXRChannelPixelData]) -> np.ndarray:
    """Extract RGB data from loaded layer channels.

    Args:
        channels: Loaded channel pixel data for a layer

    Returns:
        (H, W, 3) float32 RGB array

    Raises:
        ValueError: If no channels are provided
    """
    if not channels:
        msg = "No channels provided"
        raise ValueError(msg)
    return _channels_to_rgb(channels, strip_layer_prefix=True)


def _ensure_2d(pixels: np.ndarray) -> np.ndarray:
    """Validate that channel pixel data is 2D (H, W).

    Args:
        pixels: NumPy array to validate

    Returns:
        The same array, unchanged (for chaining)

    Raises:
        ValueError: If pixel data is not 2D
    """
    if pixels.ndim != 2:  # noqa: PLR2004
        msg = (
            f"Expected 2D channel data (H, W), got shape {pixels.shape}. "
            f"This may indicate interleaved or multi-sample data."
        )
        raise ValueError(msg)
    return pixels


def _positional_stack_rgb(pixel_arrays: list[np.ndarray]) -> np.ndarray:
    """Stack 2D pixel arrays positionally into an (H, W, 3) RGB array.

    Fallback when role-based assembly isn't possible. Uses channels in order:
    - 3+ channels: first three become R, G, B
    - 2 channels: two channels + zero-filled blue
    - 1 channel: broadcast to grayscale

    Args:
        pixel_arrays: List of 2D (H, W) float32 arrays

    Returns:
        (H, W, 3) RGB array
    """
    if len(pixel_arrays) >= 3:  # noqa: PLR2004
        return np.stack(pixel_arrays[:3], axis=-1)
    if len(pixel_arrays) == 2:  # noqa: PLR2004
        zeros = np.zeros_like(pixel_arrays[0])
        return np.stack([pixel_arrays[0], pixel_arrays[1], zeros], axis=-1)
    return np.stack([pixel_arrays[0], pixel_arrays[0], pixel_arrays[0]], axis=-1)


def _channels_to_rgb(channels: list[EXRChannelPixelData], *, strip_layer_prefix: bool = False) -> np.ndarray:
    """Assemble a (H, W, 3) RGB array from a list of loaded channels.

    Assembly strategy (first match wins):
    1. Role-based: if R, G, B roles are all present, use them
    2. Luminance: if a Y/luminance channel exists, broadcast to grayscale
    3. Positional: stack channels by position (first 3 become R, G, B)

    Args:
        channels: Channels with loaded pixel data
        strip_layer_prefix: If True, parse channel names to extract the
            short name (e.g., "beauty.R" to "R") before role matching

    Returns:
        (H, W, 3) float32 RGB array

    Raises:
        ValueError: If channels list is empty or data is not 2D
    """
    if not channels:
        msg = "No channels provided"
        raise ValueError(msg)

    normalized: list[_NormalizedChannel] = []
    for ch in channels:
        if strip_layer_prefix:
            display_name = parse_channel_name(ch.info.name).channel_name
        else:
            display_name = ch.info.name
        normalized.append(_NormalizedChannel(name=display_name, pixels=_ensure_2d(ch.pixels)))

    # Role-based assembly
    by_role: dict[str, np.ndarray] = {}
    for entry in normalized:
        role = normalize_channel_role(entry.name)
        if role and role not in by_role:
            by_role[role] = entry.pixels

    if all(role in by_role for role in ("red", "green", "blue")):
        return np.stack([by_role["red"], by_role["green"], by_role["blue"]], axis=-1)

    # Luminance as grayscale
    if "luminance" in by_role:
        lum = by_role["luminance"]
        return np.stack([lum, lum, lum], axis=-1)

    # Positional fallback
    return _positional_stack_rgb([entry.pixels for entry in normalized])


# --- Preview Generation ---


def generate_preview(  # noqa: PLR0913
    channels: list[EXRChannelPixelData],
    max_width: int = 1024,
    max_height: int = 1024,
    tone_mapping_method: ToneMappingMethod = ToneMappingMethod.SIMPLE,
    exposure: float = 0.0,
    gamma: float = 2.2,
) -> Image.Image:
    """Generate a tone-mapped preview image from loaded channel data.

    Extracts RGB, applies tone mapping, converts to 8-bit sRGB, and
    thumbnails to the max dimensions.

    Args:
        channels: Loaded channel pixel data
        max_width: Maximum preview width (aspect ratio preserved)
        max_height: Maximum preview height (aspect ratio preserved)
        tone_mapping_method: Algorithm for HDR to LDR conversion
        exposure: Exposure adjustment in stops
        gamma: Gamma correction value

    Returns:
        PIL Image with tone-mapped sRGB preview
    """
    rgb = extract_rgb_from_channels(channels)
    rgb_ldr = tone_map(rgb, tone_mapping_method, exposure, gamma)
    rgb_8bit = np.clip(rgb_ldr * 255.0, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(rgb_8bit, mode="RGB")
    pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return pil_image
