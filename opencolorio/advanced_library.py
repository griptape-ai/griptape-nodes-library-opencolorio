from __future__ import annotations

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary

from opencolorio.services.colorspace_transform import (
    ColorspaceTransformRequest,
    handle_colorspace_transform,
)
from opencolorio.services.resolve_color_params import (
    ResolveColorParamsRequest,
    handle_resolve_color_params,
)


class OpenColorIOLibrary(AdvancedNodeLibrary):
    def get_request_handlers(self) -> list:
        return [
            (ColorspaceTransformRequest, handle_colorspace_transform),
            (ResolveColorParamsRequest, handle_resolve_color_params),
        ]
