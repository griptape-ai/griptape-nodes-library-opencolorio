"""EXR data structures and I/O operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np  # noqa: TC002 - Runtime use in dataclass and method
import OpenEXR  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from griptape_nodes.retained_mode.events.os_events import ReadFileRequest


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
        channels_by_name = {ch.name: ch for ch in part.channels}
        rgb_pixels = channels_by_name["R"].pixels
    """

    channels: list[EXRChannel]
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


def attempt_read_exr(read_request: ReadFileRequest) -> EXRData:  # noqa: C901, PLR0912, PLR0915 - Complex data extraction
    """Read OpenEXR file and extract ALL data with proper structure.

    Args:
        read_request: ReadFileRequest with file path

    Returns:
        EXRData containing list of EXRPart objects with structured headers

    Raises:
        ValueError: If request has no file path, EXR has no channels, or unknown enum values
        RuntimeError: If EXR loading fails (includes FileNotFoundError from OpenEXR)

    Note:
        Supports multi-part EXR files. Each part is returned as separate EXRPart.
        TODO: Unify path extraction logic with OSManager.on_read_file_request (lines 1579-1590)
    """
    # FAILURES FIRST - Get file path from request (duplicate logic from os_manager.py:1579-1590)
    if read_request.file_entry is not None:
        file_path = read_request.file_entry.path
    elif read_request.file_path is not None:
        file_path = read_request.file_path
    else:
        msg = "ReadFileRequest must have either file_path or file_entry"
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
                            x_sampling = int(ch_str.split("xSampling=")[1].split(",")[0].split(")")[0])
                        if "ySampling=" in ch_str:
                            y_sampling = int(ch_str.split("ySampling=")[1].split(")")[0])

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

            # SUCCESS PATH - Return everything extracted
            return EXRData(parts=exr_parts)

    except ValueError:
        raise
    except Exception as e:
        msg = f"Failed to load EXR file '{file_path}': {e}"
        raise RuntimeError(msg) from e
