"""EXR file artifact — descriptor for an entire EXR file.

Carries file path + part-level metadata for all parts/subimages without
loading pixel data. Downstream nodes use this to understand the full file
structure and then load specific channels via OIIO.
"""

from __future__ import annotations

from dataclasses import dataclass

from griptape_nodes_opencolorio.exr.exr_part_artifact import EXRPartArtifact


@dataclass
class EXRFileArtifact:
    """Descriptor for an EXR file. Path + header metadata, no pixel data.

    This is the primary output of the LoadEXR node's file-level output.
    Downstream nodes use this to understand the file structure and then
    call load_channels() to get specific pixel data.

    Attributes:
        file_path: Absolute path to the EXR file on disk
        parts: Self-contained descriptors for each part/subimage
    """

    file_path: str
    parts: list[EXRPartArtifact]

    def to_text(self) -> str:
        total_channels = sum(len(p.channels) for p in self.parts)
        total_layers = sum(len(p.layers) for p in self.parts)
        return (
            f"EXR: {self.file_path} "
            f"({len(self.parts)} parts, {total_layers} layers, {total_channels} channels)"
        )
