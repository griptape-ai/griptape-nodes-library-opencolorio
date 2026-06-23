from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from opencolorio.services.colorspace_transform import (
    ColorspaceTransformRequest,
    ColorspaceTransformResultFailure,
    ColorspaceTransformResultSuccess,
    handle_colorspace_transform,
)


def _pixels(h: int = 4, w: int = 4) -> np.ndarray:
    return np.random.default_rng(0).random((h, w, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# ColorspaceTransformResultSuccess unit tests
# ---------------------------------------------------------------------------


class TestColorspaceTransformResultSuccess:
    def test_succeeded_returns_true(self) -> None:
        result = ColorspaceTransformResultSuccess(pixels=_pixels(), result_details="ok")
        assert result.succeeded() is True

    def test_workflow_not_altered(self) -> None:
        result = ColorspaceTransformResultSuccess(pixels=_pixels(), result_details="ok")
        assert result.altered_workflow_state is False


# ---------------------------------------------------------------------------
# ColorspaceTransformResultFailure unit tests
# ---------------------------------------------------------------------------


class TestColorspaceTransformResultFailure:
    def test_succeeded_returns_false(self) -> None:
        result = ColorspaceTransformResultFailure(result_details="something went wrong")
        assert result.succeeded() is False

    def test_workflow_not_altered(self) -> None:
        result = ColorspaceTransformResultFailure(result_details="something went wrong")
        assert result.altered_workflow_state is False


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------


def _make_request(**kwargs) -> ColorspaceTransformRequest:
    defaults = {
        "pixels": _pixels(),
        "source_colorspace": "scene_linear",
        "config_path": None,
        "display": "sRGB",
        "view": "ACES",
    }
    defaults.update(kwargs)
    return ColorspaceTransformRequest(**defaults)


def _mock_cpu_processor():
    proc = MagicMock()
    proc.applyRGB = MagicMock()
    return proc


def _mock_ocio_config(cpu_processor=None):
    if cpu_processor is None:
        cpu_processor = _mock_cpu_processor()
    config = MagicMock()
    config.getProcessor.return_value.getDefaultCPUProcessor.return_value = cpu_processor
    return config, cpu_processor


class TestHandleColorspaceTransform:
    def test_returns_success_with_pixels(self) -> None:
        config, proc = _mock_ocio_config()
        req = _make_request()

        with patch("opencolorio.services.colorspace_transform.load_ocio_config", return_value=config):
            result = handle_colorspace_transform(req)

        assert isinstance(result, ColorspaceTransformResultSuccess)
        assert result.succeeded() is True
        assert result.pixels is not None
        assert result.pixels.shape == req.pixels.shape

    def test_does_not_mutate_input_pixels(self) -> None:
        config, proc = _mock_ocio_config()
        original = _pixels()
        req = _make_request(pixels=original.copy())
        input_copy = original.copy()

        with patch("opencolorio.services.colorspace_transform.load_ocio_config", return_value=config):
            handle_colorspace_transform(req)

        np.testing.assert_array_equal(req.pixels, input_copy)

    def test_display_view_path_calls_get_processor_with_transform(self) -> None:
        config, proc = _mock_ocio_config()
        req = _make_request(display="sRGB", view="ACES")

        with patch("opencolorio.services.colorspace_transform.load_ocio_config", return_value=config):
            result = handle_colorspace_transform(req)

        config.getProcessor.assert_called_once()
        assert result.succeeded() is True

    def test_no_display_view_returns_passthrough(self) -> None:
        req = _make_request(display="", view="")
        original = req.pixels.copy()

        # No OCIO config needed for passthrough — must not call load_ocio_config
        with patch("opencolorio.services.colorspace_transform.load_ocio_config") as mock_load:
            result = handle_colorspace_transform(req)
            mock_load.assert_not_called()

        assert isinstance(result, ColorspaceTransformResultSuccess)
        assert result.succeeded() is True
        np.testing.assert_array_equal(result.pixels, original)

    def test_returns_failure_on_bad_config(self) -> None:
        req = _make_request(config_path="/nonexistent/config.ocio")

        with patch(
            "opencolorio.services.colorspace_transform.load_ocio_config",
            side_effect=Exception("config not found"),
        ):
            result = handle_colorspace_transform(req)

        assert isinstance(result, ColorspaceTransformResultFailure)
        assert result.succeeded() is False
        assert "config not found" in str(result.result_details)
        assert "/nonexistent/config.ocio" in str(result.result_details)

    def test_returns_failure_on_processor_error(self) -> None:
        config = MagicMock()
        config.getProcessor.side_effect = Exception("unknown colorspace 'bad_cs'")
        req = _make_request(source_colorspace="bad_cs")

        with patch("opencolorio.services.colorspace_transform.load_ocio_config", return_value=config):
            result = handle_colorspace_transform(req)

        assert isinstance(result, ColorspaceTransformResultFailure)
        assert result.succeeded() is False
        assert "bad_cs" in str(result.result_details)

    def test_apply_rgb_called_on_output_copy(self) -> None:
        config, proc = _mock_ocio_config()
        pixels = _pixels()
        req = _make_request(pixels=pixels)

        with patch("opencolorio.services.colorspace_transform.load_ocio_config", return_value=config):
            result = handle_colorspace_transform(req)

        assert isinstance(result, ColorspaceTransformResultSuccess)
        proc.applyRGB.assert_called_once()
        # The arg passed to applyRGB should be the output (not the original)
        called_arg = proc.applyRGB.call_args[0][0]
        assert called_arg is result.pixels

    def test_output_pixels_are_float32(self) -> None:
        config, proc = _mock_ocio_config()
        req = _make_request()

        with patch("opencolorio.services.colorspace_transform.load_ocio_config", return_value=config):
            result = handle_colorspace_transform(req)

        assert isinstance(result, ColorspaceTransformResultSuccess)
        assert result.pixels is not None
        assert result.pixels.dtype == np.float32
