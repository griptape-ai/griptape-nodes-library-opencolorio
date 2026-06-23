from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import PyOpenColorIO as ocio
from griptape_nodes.retained_mode.events.base_events import RequestPayload, ResultDetails, ResultPayload

from opencolorio.ocio_helpers import load_ocio_config


@dataclass(kw_only=True)
class ColorspaceTransformRequest(RequestPayload):
    """Request a display-view colour transform via OCIO.

    When *display* and *view* are both set, a DisplayViewTransform is applied.
    When both are empty the pixels are returned as-is (passthrough).
    *broadcast_result* defaults to False because pixel buffers are large.
    """

    pixels: np.ndarray
    source_colorspace: str
    config_path: str = ""
    display: str = ""
    view: str = ""
    broadcast_result: bool = False


@dataclass(kw_only=True)
class ColorspaceTransformResult(ResultPayload):
    """Result of a ColorspaceTransformRequest."""

    pixels: np.ndarray | None = None
    result_details: ResultDetails | str = ""

    def succeeded(self) -> bool:
        return self.pixels is not None


def handle_colorspace_transform(req: ColorspaceTransformRequest) -> ColorspaceTransformResult:
    try:
        if not (req.display and req.view):
            return ColorspaceTransformResult(pixels=req.pixels.copy())

        config = load_ocio_config(req.config_path)
        transform = ocio.DisplayViewTransform(
            src=req.source_colorspace,
            display=req.display,
            view=req.view,
        )
        cpu_processor = config.getProcessor(transform).getDefaultCPUProcessor()
        out = req.pixels.copy()
        cpu_processor.applyRGB(out)
        return ColorspaceTransformResult(
            pixels=out,
            result_details=f"Transformed {req.source_colorspace!r} → {req.display}/{req.view}",
        )
    except Exception as e:
        return ColorspaceTransformResult(pixels=None, result_details=str(e))
