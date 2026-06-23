from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import PyOpenColorIO as ocio
from griptape_nodes.retained_mode.events.base_events import (
    RequestPayload,
    ResultPayloadFailure,
    ResultPayloadSuccess,
    WorkflowNotAlteredMixin,
)

from opencolorio.ocio_helpers import load_ocio_config


@dataclass(kw_only=True)
class ColorspaceTransformRequest(RequestPayload):
    """Request a display-view colour transform via OCIO.

    Args:
        pixels: Input image data as an (H, W, 3) float32 array.
        source_colorspace: OCIO colorspace name of the input pixels.
        config_path: Path to an OCIO config file. None or empty uses OCIO_CONFIG from the environment.
        display: OCIO display device name. Empty string disables the transform (passthrough).
        view: OCIO view name. Empty string disables the transform (passthrough).
        broadcast_result: Whether to broadcast the result. Defaults to False because pixel buffers are large.
    """

    pixels: np.ndarray
    source_colorspace: str
    config_path: str | None = None
    display: str = ""
    view: str = ""
    broadcast_result: bool = False


@dataclass(kw_only=True)
class ColorspaceTransformResultSuccess(WorkflowNotAlteredMixin, ResultPayloadSuccess):
    """Successful colorspace transform result."""

    pixels: np.ndarray


@dataclass(kw_only=True)
class ColorspaceTransformResultFailure(WorkflowNotAlteredMixin, ResultPayloadFailure):
    """Failed colorspace transform result."""


def handle_colorspace_transform(
    req: ColorspaceTransformRequest,
) -> ColorspaceTransformResultSuccess | ColorspaceTransformResultFailure:
    if not (req.display and req.view):
        return ColorspaceTransformResultSuccess(
            pixels=req.pixels.copy(),
            result_details="Passthrough: no display/view specified.",
        )

    try:
        config = load_ocio_config(req.config_path)
    except Exception as e:
        return ColorspaceTransformResultFailure(
            result_details=(f"Attempted to load OCIO config with path {req.config_path!r}. Failed due to: {e}"),
            exception=e,
        )

    try:
        transform = ocio.DisplayViewTransform(
            src=req.source_colorspace,
            display=req.display,
            view=req.view,
        )
        cpu_processor = config.getProcessor(transform).getDefaultCPUProcessor()
        out = req.pixels.copy()
        cpu_processor.applyRGB(out)
        return ColorspaceTransformResultSuccess(
            pixels=out,
            result_details=f"Transformed {req.source_colorspace!r} → {req.display}/{req.view}",
        )
    except Exception as e:
        return ColorspaceTransformResultFailure(
            result_details=(
                f"Attempted to transform colorspace {req.source_colorspace!r} "
                f"to display={req.display!r}/view={req.view!r}. Failed due to: {e}"
            ),
            exception=e,
        )
