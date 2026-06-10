from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class OCIOConfigArtifact:
    """Descriptor for an OpenColorIO configuration.

    Flows between OCIO nodes. Downstream nodes call
    ``ocio.Config.CreateFromFile(artifact.file_path)`` (or ``CreateFromEnv()``
    when ``file_path`` is empty) and build a context from ``context_vars``.
    """

    file_path: str  # absolute path, or "" to use $OCIO env var
    context_vars: dict[str, str] = field(default_factory=dict)

    def to_text(self) -> str:
        src = self.file_path or "$OCIO"
        ctx = f" ctx={self.context_vars}" if self.context_vars else ""
        return f"OCIOConfigArtifact({src}{ctx})"

    def __str__(self) -> str:
        # Return JSON so that the standard ToJson node can deserialize this artifact.
        return json.dumps(dataclasses.asdict(self))
