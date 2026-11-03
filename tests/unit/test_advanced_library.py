from __future__ import annotations

from opencolorio.advanced_library import OpenColorIOLibrary
from opencolorio.services.colorspace_transform import (
    ColorspaceTransformRequest,
    handle_colorspace_transform,
)


class TestOpenColorIOLibrary:
    def test_get_request_handlers_returns_a_pair_per_service(self) -> None:
        handlers = OpenColorIOLibrary().get_request_handlers()
        assert len(handlers) == 2

    def test_colorspace_transform_request_is_registered(self) -> None:
        handlers = dict(OpenColorIOLibrary().get_request_handlers())
        assert ColorspaceTransformRequest in handlers

    def test_every_handler_is_callable(self) -> None:
        assert all(callable(handler) for _, handler in OpenColorIOLibrary().get_request_handlers())

    def test_colorspace_transform_maps_to_its_own_handler(self) -> None:
        handlers = dict(OpenColorIOLibrary().get_request_handlers())
        assert handlers[ColorspaceTransformRequest] is handle_colorspace_transform

    def test_each_request_type_is_registered_once(self) -> None:
        # The engine raises if a request type already has a handler.
        handlers = OpenColorIOLibrary().get_request_handlers()
        assert len({request_type for request_type, _ in handlers}) == len(handlers)
