"""EXR part artifact — descriptor for a single part within an EXR file.

Self-contained: carries file path + part index + full header/channel/layer
metadata. Downstream nodes can use this to load any channel from the part
via load_channels().
"""

from __future__ import annotations

from dataclasses import dataclass

from griptape_nodes_opencolorio.exr.exr import EXRChannelInfo, EXRHeader, EXRLayer


@dataclass
class EXRPartArtifact:
    """Descriptor for a single part within an EXR file.

    Self-contained: a downstream node receiving this has everything it needs
    to understand the part's structure and load any channel via
    load_channels(file_path, part_index, channel_indices).

    Attributes:
        file_path: Absolute path to the EXR file on disk
        part_index: Part index (0-based)
        name: Part name (empty string for single-part or unnamed parts)
        width: Image width in pixels
        height: Image height in pixels
        header: Full EXR header metadata
        channels: Channel metadata (no pixel data)
        layers: Layer groupings derived from channel names
    """

    file_path: str
    part_index: int
    name: str
    width: int
    height: int
    header: EXRHeader
    channels: list[EXRChannelInfo]
    layers: list[EXRLayer]

    def to_text(self) -> str:
        display_name = self.name or f"part {self.part_index}"
        return (
            f"EXR Part '{display_name}' from {self.file_path} "
            f"({len(self.layers)} layers, {len(self.channels)} channels, "
            f"{self.width}x{self.height})"
        )
