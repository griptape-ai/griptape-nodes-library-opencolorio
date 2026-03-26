"""EXR layer artifact — descriptor for a single layer within an EXR part.

Composes an EXRPartArtifact (which part) with an EXRLayer (which layer).
Downstream nodes use this to load exactly the channels they need via
load_channels(artifact.part.file_path, artifact.part.part_index, channel_indices).
"""

from __future__ import annotations

from dataclasses import dataclass

from griptape_nodes_opencolorio.exr.exr import EXRLayer
from griptape_nodes_opencolorio.exr.exr_part_artifact import EXRPartArtifact


@dataclass
class EXRLayerArtifact:
    """Descriptor for a single layer within an EXR part.

    Composes a part artifact (file path, part index, header, all channels)
    with an EXR layer (the specific channel subset for this layer).

    Attributes:
        part: The part this layer belongs to
        layer: Channel metadata for this specific layer
    """

    part: EXRPartArtifact
    layer: EXRLayer

    def to_text(self) -> str:
        display_name = self.layer.name or "default"
        return (
            f"EXR Layer '{display_name}' from {self.part.file_path} "
            f"(part {self.part.part_index}, {len(self.layer.channels)} channels, "
            f"{self.part.width}x{self.part.height})"
        )
