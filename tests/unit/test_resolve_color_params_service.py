"""Tests for the ResolveColorParamsRequest handler.

The point of this request is that a library which has never imported this
package can ask for resolved colour parameters, so several tests here are about
what the request is made of rather than what it returns.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from opencolorio.advanced_library import OpenColorIOLibrary
from opencolorio.defaults import DEFAULT_SENTINEL
from opencolorio.services.resolve_color_params import (
    ResolveColorParamsRequest,
    ResolveColorParamsResultFailure,
    ResolveColorParamsResultSuccess,
    handle_resolve_color_params,
)

CG_V4_URI = "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5"
CG_V1_URI = "ocio://cg-config-v1.0.0_aces-v1.3_ocio-v2.1"


class TestResultPayloads:
    def test_success_succeeded(self) -> None:
        result = ResolveColorParamsResultSuccess(
            working_space="ACEScg",
            source_colorspace="ACEScg",
            display="sRGB - Display",
            view="Raw",
            config_path=None,
            sources={},
            result_details="ok",
        )
        assert result.succeeded() is True

    def test_success_does_not_alter_the_workflow(self) -> None:
        result = ResolveColorParamsResultSuccess(
            working_space="",
            source_colorspace="",
            display="",
            view="",
            config_path=None,
            sources={},
            result_details="ok",
        )
        assert result.altered_workflow_state is False

    def test_failure_did_not_succeed(self) -> None:
        assert ResolveColorParamsResultFailure(result_details="boom").succeeded() is False


class TestRequestShape:
    def test_it_is_constructible_with_no_arguments(self) -> None:
        assert ResolveColorParamsRequest() is not None

    def test_it_is_constructible_from_primitives_alone(self) -> None:
        # A library that discovered this type by name has never imported our
        # artifact classes and could not construct them.
        request = ResolveColorParamsRequest(
            config_path="/studio/show.ocio",
            context_vars={"SHOT": "sh010"},
            file_path="/shots/sh010/plate.exr",
            explicit_source_colorspace="ACEScg",
            explicit_display="sRGB - Display",
            explicit_view="Raw",
        )
        assert request.config_path == "/studio/show.ocio"

    def test_context_vars_defaults_to_an_empty_dict(self) -> None:
        assert ResolveColorParamsRequest().context_vars == {}

    def test_two_requests_do_not_share_a_context_vars_dict(self) -> None:
        first = ResolveColorParamsRequest()
        first.context_vars["SHOT"] = "sh010"
        assert ResolveColorParamsRequest().context_vars == {}


class TestHandler:
    def test_an_explicit_config_path_is_used(self) -> None:
        result = handle_resolve_color_params(ResolveColorParamsRequest(config_path=CG_V1_URI))

        assert isinstance(result, ResolveColorParamsResultSuccess)
        assert result.config_path == CG_V1_URI
        assert result.working_space == "ACEScg"

    def test_no_config_path_runs_the_discovery_chain(self) -> None:
        # With $OCIO unset and no fallback configured, this still answers.
        result = handle_resolve_color_params(ResolveColorParamsRequest())

        assert isinstance(result, ResolveColorParamsResultSuccess)
        assert result.sources

    def test_no_config_path_follows_ocio_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OCIO", CG_V4_URI)

        result = handle_resolve_color_params(ResolveColorParamsRequest())

        assert isinstance(result, ResolveColorParamsResultSuccess)
        assert result.display == "sRGB - Display"

    def test_explicit_values_win(self) -> None:
        result = handle_resolve_color_params(
            ResolveColorParamsRequest(config_path=CG_V4_URI, explicit_display="Rec.1886 Rec.709 - Display")
        )

        assert isinstance(result, ResolveColorParamsResultSuccess)
        assert result.display == "Rec.1886 Rec.709 - Display"
        assert result.sources["display"] == "explicit"

    def test_the_sentinel_counts_as_not_explicit(self) -> None:
        # A consumer mirroring the node's dropdowns should not have to translate.
        result = handle_resolve_color_params(
            ResolveColorParamsRequest(config_path=CG_V4_URI, explicit_display=DEFAULT_SENTINEL)
        )

        assert isinstance(result, ResolveColorParamsResultSuccess)
        assert result.display == "sRGB - Display"
        assert result.sources["display"] != "explicit"

    def test_file_path_drives_the_file_rules(self) -> None:
        result = handle_resolve_color_params(
            ResolveColorParamsRequest(config_path=CG_V4_URI, file_path="/shots/sh010/plate.exr")
        )

        assert isinstance(result, ResolveColorParamsResultSuccess)
        assert result.source_colorspace == "ACES2065-1"
        assert result.sources["source_colorspace"] == "config-file-rule"

    def test_the_result_carries_provenance_for_every_slot(self) -> None:
        result = handle_resolve_color_params(ResolveColorParamsRequest(config_path=CG_V4_URI))

        assert isinstance(result, ResolveColorParamsResultSuccess)
        assert set(result.sources) == {"working_space", "source_colorspace", "display", "view"}

    def test_a_bad_config_returns_a_failure_rather_than_raising(self) -> None:
        result = handle_resolve_color_params(ResolveColorParamsRequest(config_path="/nonexistent/nope.ocio"))

        assert isinstance(result, ResolveColorParamsResultFailure)

    def test_the_failure_names_the_config(self) -> None:
        result = handle_resolve_color_params(ResolveColorParamsRequest(config_path="/nonexistent/nope.ocio"))

        assert "/nonexistent/nope.ocio" in str(result.result_details)

    def test_warnings_reach_result_details(self) -> None:
        # .png on this config resolves to a display-referred space.
        result = handle_resolve_color_params(
            ResolveColorParamsRequest(config_path=CG_V4_URI, file_path="/refs/board.png")
        )

        assert isinstance(result, ResolveColorParamsResultSuccess)
        assert "display-referred" in str(result.result_details)

    def test_context_vars_are_accepted_and_do_not_change_resolution(self) -> None:
        without = handle_resolve_color_params(ResolveColorParamsRequest(config_path=CG_V4_URI))
        with_vars = handle_resolve_color_params(
            ResolveColorParamsRequest(config_path=CG_V4_URI, context_vars={"SHOT": "sh010"})
        )

        assert isinstance(without, ResolveColorParamsResultSuccess)
        assert isinstance(with_vars, ResolveColorParamsResultSuccess)
        assert without.display == with_vars.display


class TestRegistration:
    def test_both_handlers_are_registered(self) -> None:
        assert len(OpenColorIOLibrary().get_request_handlers()) == 2

    def test_the_resolve_request_is_registered(self) -> None:
        handlers = dict(OpenColorIOLibrary().get_request_handlers())
        assert ResolveColorParamsRequest in handlers

    def test_it_maps_to_its_own_handler(self) -> None:
        handlers = dict(OpenColorIOLibrary().get_request_handlers())
        assert handlers[ResolveColorParamsRequest] is handle_resolve_color_params

    def test_it_is_discoverable_by_name(self) -> None:
        # This is how a consuming library finds it: match on __name__, with no
        # import of this package anywhere.
        names = [request_type.__name__ for request_type, _ in OpenColorIOLibrary().get_request_handlers()]
        assert "ResolveColorParamsRequest" in names

    def test_a_consumer_can_build_it_from_the_discovered_type(self) -> None:
        discovered = next(
            request_type
            for request_type, _ in OpenColorIOLibrary().get_request_handlers()
            if request_type.__name__ == "ResolveColorParamsRequest"
        )

        request = discovered(config_path=CG_V4_URI, file_path="/shots/sh010/plate.exr")
        result = handle_resolve_color_params(request)

        assert isinstance(result, ResolveColorParamsResultSuccess)


class TestSettingsLayer:
    def test_a_settings_override_is_honoured(self) -> None:
        from griptape_nodes.retained_mode.events.config_events import (
            GetConfigValueResultFailure,
            GetConfigValueResultSuccess,
        )

        def handle_request(request: object) -> object:
            key = str(getattr(request, "category_and_key", ""))
            if key == "opencolorio.default_display":
                return GetConfigValueResultSuccess(value="Rec.1886 Rec.709 - Display", result_details="ok")
            return GetConfigValueResultFailure(result_details="not set")

        mock = MagicMock()
        mock.GriptapeNodes.handle_request.side_effect = handle_request
        with patch("opencolorio.defaults.griptape_nodes", mock):
            result = handle_resolve_color_params(ResolveColorParamsRequest(config_path=CG_V4_URI))

        assert isinstance(result, ResolveColorParamsResultSuccess)
        assert result.display == "Rec.1886 Rec.709 - Display"
        assert result.sources["display"] == "settings"
