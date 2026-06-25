from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass


@dataclass
class OCIOColorParamsArtifact:
    """Bundled colorspace transform parameters for wiring to downstream nodes.

    Carries the three values needed by DisplayEXRPart (and similar nodes) so
    that a single OCIOColorParameters node can drive multiple consumers.
    """

    source_colorspace: str
    display: str
    view: str

    def to_text(self) -> str:
        return (
            f"OCIOColorParamsArtifact(source={self.source_colorspace!r}, display={self.display!r}, view={self.view!r})"
        )

    def __str__(self) -> str:
        return json.dumps(dataclasses.asdict(self))
