from __future__ import annotations

import json

from opencolorio.artifacts.ocio_color_params_artifact import OCIOColorParamsArtifact


class TestOCIOColorParamsArtifactToText:
    def test_to_text_format(self) -> None:
        artifact = OCIOColorParamsArtifact(
            source_colorspace="ACEScg",
            display="ACES",
            view="sRGB",
        )
        assert artifact.to_text() == "OCIOColorParamsArtifact(source='ACEScg', display='ACES', view='sRGB')"

    def test_to_text_empty_strings(self) -> None:
        artifact = OCIOColorParamsArtifact(source_colorspace="", display="", view="")
        result = artifact.to_text()
        assert result.startswith("OCIOColorParamsArtifact(")
        assert "source=''" in result


class TestOCIOColorParamsArtifactStr:
    def test_str_is_valid_json(self) -> None:
        artifact = OCIOColorParamsArtifact(
            source_colorspace="ACEScg",
            display="ACES",
            view="sRGB",
        )
        parsed = json.loads(str(artifact))
        assert parsed == {"source_colorspace": "ACEScg", "display": "ACES", "view": "sRGB"}

    def test_str_round_trips(self) -> None:
        artifact = OCIOColorParamsArtifact(
            source_colorspace="scene_linear",
            display="sRGB",
            view="Film",
        )
        data = json.loads(str(artifact))
        assert OCIOColorParamsArtifact(**data) == artifact

    def test_str_empty_strings_valid_json(self) -> None:
        artifact = OCIOColorParamsArtifact(source_colorspace="", display="", view="")
        parsed = json.loads(str(artifact))
        assert parsed == {"source_colorspace": "", "display": "", "view": ""}


class TestOCIOColorParamsArtifactEquality:
    def test_equal_instances(self) -> None:
        a = OCIOColorParamsArtifact(source_colorspace="ACEScg", display="ACES", view="sRGB")
        b = OCIOColorParamsArtifact(source_colorspace="ACEScg", display="ACES", view="sRGB")
        assert a == b

    def test_unequal_instances(self) -> None:
        a = OCIOColorParamsArtifact(source_colorspace="ACEScg", display="ACES", view="sRGB")
        b = OCIOColorParamsArtifact(source_colorspace="scene_linear", display="ACES", view="sRGB")
        assert a != b
