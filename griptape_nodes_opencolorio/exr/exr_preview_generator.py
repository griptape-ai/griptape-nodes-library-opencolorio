"""EXR preview generation with HDR tone mapping utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

from griptape_nodes_opencolorio.exr.exr import EXRData, EXRPart  # noqa: TC001


def apply_exposure(rgb: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    """Apply exposure adjustment in stops.

    Args:
        rgb: RGB image data as NumPy array
        exposure: Exposure adjustment in stops (-10 to +10)

    Returns:
        Exposure-adjusted RGB data
    """
    exposure_multiplier = 2.0**exposure
    return rgb * exposure_multiplier


def apply_gamma(rgb: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Apply gamma correction.

    Args:
        rgb: RGB image data as NumPy array
        gamma: Gamma correction value (0.1 to 5.0)

    Returns:
        Gamma-corrected RGB data
    """
    # Clamp to avoid negative values in gamma correction
    rgb_clamped = np.maximum(rgb, 0.0)
    return np.power(rgb_clamped, 1.0 / gamma)


def tone_map_simple(rgb: np.ndarray, exposure: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """Simple tone mapping: exposure + clamp + gamma.

    Args:
        rgb: HDR RGB image data as NumPy array
        exposure: Exposure adjustment in stops
        gamma: Gamma correction value

    Returns:
        Tone-mapped RGB data in [0, 1] range
    """
    # Apply exposure
    rgb = apply_exposure(rgb, exposure)

    # Clamp to [0, 1]
    rgb = np.clip(rgb, 0.0, 1.0)

    # Apply gamma
    rgb = apply_gamma(rgb, gamma)

    return rgb


def tone_map_reinhard(rgb: np.ndarray, exposure: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """Reinhard tone mapping operator.

    Args:
        rgb: HDR RGB image data as NumPy array
        exposure: Exposure adjustment in stops
        gamma: Gamma correction value

    Returns:
        Tone-mapped RGB data in [0, 1] range
    """
    # Apply exposure first
    rgb = apply_exposure(rgb, exposure)

    # Reinhard formula: L / (1 + L)
    rgb_mapped = rgb / (1.0 + rgb)

    # Apply gamma
    rgb_mapped = apply_gamma(rgb_mapped, gamma)

    return rgb_mapped


def tone_map_filmic(rgb: np.ndarray, exposure: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """Filmic tone mapping (Blender-style approximation).

    Based on John Hable's Uncharted 2 tone mapping.

    Args:
        rgb: HDR RGB image data as NumPy array
        exposure: Exposure adjustment in stops
        gamma: Gamma correction value

    Returns:
        Tone-mapped RGB data in [0, 1] range
    """
    # Apply exposure first
    rgb = apply_exposure(rgb, exposure)

    # Filmic curve approximation
    x = rgb
    a = 0.22  # Shoulder strength
    b = 0.30  # Linear strength
    c = 0.10  # Linear angle
    d = 0.20  # Toe strength
    e = 0.01  # Toe numerator
    f = 0.30  # Toe denominator

    rgb_mapped = ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f

    # Normalize
    white_point = 11.2
    white_mapped = (
        (white_point * (a * white_point + c * b) + d * e) / (white_point * (a * white_point + b) + d * f)
    ) - e / f
    rgb_mapped = rgb_mapped / white_mapped

    # Apply gamma
    rgb_mapped = apply_gamma(rgb_mapped, gamma)

    return rgb_mapped


def tone_map(
    rgb: np.ndarray,
    method: Literal["simple", "reinhard", "filmic"] = "simple",
    exposure: float = 0.0,
    gamma: float = 2.2,
) -> np.ndarray:
    """Apply tone mapping based on selected algorithm.

    Args:
        rgb: HDR RGB image data as NumPy array
        method: Tone mapping algorithm to use
        exposure: Exposure adjustment in stops (-10 to +10)
        gamma: Gamma correction value (0.1 to 5.0)

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


def extract_rgb_from_exr_part(part: EXRPart) -> np.ndarray:
    """Extract RGB data from an EXR part.

    Tries to find RGB channels in this order:
    1. Interleaved "RGB" channel
    2. Separate "R", "G", "B" channels
    3. First available channel as grayscale

    Args:
        part: EXRPart containing channel data

    Returns:
        RGB data as NumPy array (H, W, 3)

    Raises:
        ValueError: If no channels are available
    """
    channels = {ch.name: ch for ch in part.channels}

    # Try to get RGB channels
    if "RGB" in channels:
        # Interleaved RGB channel
        return channels["RGB"].pixels  # Should already be (H, W, 3)
    if all(ch in channels for ch in ["R", "G", "B"]):
        # Separate R, G, B channels
        r = channels["R"].pixels  # (H, W)
        g = channels["G"].pixels
        b = channels["B"].pixels
        return np.stack([r, g, b], axis=-1)

    # Fallback: use first available channel as grayscale
    if not channels:
        msg = "EXR part has no channels"
        raise ValueError(msg)
    first_channel_name = next(iter(channels.keys()))
    gray = channels[first_channel_name].pixels
    return np.stack([gray, gray, gray], axis=-1)


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
        tone_mapping_method: Tone mapping algorithm to use
        exposure: Exposure adjustment in stops
        gamma: Gamma correction value
        part_index: Which part to use for preview (for multi-part EXRs)

    Returns:
        PIL Image with tone-mapped preview

    Raises:
        ValueError: If part_index is out of range or no channels available
    """
    if part_index >= len(exr_data.parts):
        msg = f"Part index {part_index} out of range (EXR has {len(exr_data.parts)} parts)"
        raise ValueError(msg)

    part = exr_data.parts[part_index]

    # Extract RGB data from the part
    rgb = extract_rgb_from_exr_part(part)

    # Apply tone mapping
    rgb_ldr = tone_map(rgb, tone_mapping_method, exposure, gamma)

    # Clamp and convert to 8-bit
    rgb_8bit = np.clip(rgb_ldr * 255.0, 0, 255).astype(np.uint8)

    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_8bit, mode="RGB")

    # Resize to thumbnail dimensions if needed
    pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

    return pil_image
