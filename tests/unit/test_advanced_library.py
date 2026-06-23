from __future__ import annotations

from opencolorio.advanced_library import OpenColorIOLibrary
from opencolorio.services.colorspace_transform import (
    ColorspaceTransformRequest,
    handle_colorspace_transform,
)


class TestOpenColorIOLibrary:
    def test_get_request_handlers_returns_one_pair(self) -> None:
        handlers = OpenColorIOLibrary().get_request_handlers()
        assert len(handlers) == 1

    def test_request_type_is_colorspace_transform_request(self) -> None:
        request_type, _ = OpenColorIOLibrary().get_request_handlers()[0]
        assert request_type is ColorspaceTransformRequest

    def test_handler_is_callable(self) -> None:
        _, handler = OpenColorIOLibrary().get_request_handlers()[0]
        assert callable(handler)

    def test_handler_is_handle_colorspace_transform(self) -> None:
        _, handler = OpenColorIOLibrary().get_request_handlers()[0]
        assert handler is handle_colorspace_transform
