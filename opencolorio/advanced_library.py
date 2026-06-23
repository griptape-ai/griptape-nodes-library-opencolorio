from __future__ import annotations

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary

from opencolorio.services.colorspace_transform import (
    ColorspaceTransformRequest,
    handle_colorspace_transform,
)


class OpenColorIOLibrary(AdvancedNodeLibrary):
    def get_request_handlers(self) -> list:
        return [(ColorspaceTransformRequest, handle_colorspace_transform)]
