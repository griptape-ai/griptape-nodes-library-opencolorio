from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass


@dataclass
class OCIOColorParamsArtifact:
    """Bundled colorspace transform parameters for wiring to downstream nodes.

    Carries the values needed by DisplayEXRPart (and similar nodes) so
    that a single OCIOColorParameters node can drive multiple consumers.
    """

    source_colorspace: str
    display: str
    view: str
    config_path: str | None = None

    def to_text(self) -> str:
        config = self.config_path or "$OCIO"
        return (
            f"OCIOColorParamsArtifact(source={self.source_colorspace!r}, display={self.display!r},"
            f" view={self.view!r}, config={config})"
        )

    def __str__(self) -> str:
        return json.dumps(dataclasses.asdict(self))
