from __future__ import annotations

from opencolorio.artifacts.ocio_config_artifact import OCIOConfigArtifact


class TestOCIOConfigArtifactToText:
    def test_to_text_with_file_path(self) -> None:
        artifact = OCIOConfigArtifact(file_path="/path/to/config.ocio")
        assert artifact.to_text() == "OCIOConfigArtifact(/path/to/config.ocio)"

    def test_to_text_none_path_uses_env_var(self) -> None:
        artifact = OCIOConfigArtifact(file_path=None)
        assert artifact.to_text() == "OCIOConfigArtifact($OCIO)"

    def test_to_text_with_context_vars(self) -> None:
        artifact = OCIOConfigArtifact(file_path=None, context_vars={"SHOT": "sh010"})
        result = artifact.to_text()
        assert "ctx=" in result
        assert "SHOT" in result

    def test_to_text_no_context_vars_no_suffix(self) -> None:
        artifact = OCIOConfigArtifact(file_path="/cfg.ocio")
        assert "ctx=" not in artifact.to_text()

    def test_to_text_file_path_and_context_vars(self) -> None:
        artifact = OCIOConfigArtifact(file_path="/cfg.ocio", context_vars={"SEQ": "sq020"})
        result = artifact.to_text()
        assert result.startswith("OCIOConfigArtifact(/cfg.ocio")
        assert "SEQ" in result


class TestOCIOConfigArtifactStr:
    def test_str_is_valid_json(self) -> None:
        import json

        artifact = OCIOConfigArtifact(file_path="/cfg.ocio")
        parsed = json.loads(str(artifact))
        assert parsed == {"file_path": "/cfg.ocio", "context_vars": {}}

    def test_str_none_path_is_valid_json(self) -> None:
        import json

        artifact = OCIOConfigArtifact(file_path=None)
        parsed = json.loads(str(artifact))
        assert parsed == {"file_path": None, "context_vars": {}}

    def test_str_with_context_vars_is_valid_json(self) -> None:
        import json

        artifact = OCIOConfigArtifact(file_path="/cfg.ocio", context_vars={"SHOT": "sh010"})
        parsed = json.loads(str(artifact))
        assert parsed == {"file_path": "/cfg.ocio", "context_vars": {"SHOT": "sh010"}}

    def test_to_text_still_human_readable(self) -> None:
        artifact = OCIOConfigArtifact(file_path="/cfg.ocio")
        assert artifact.to_text() == "OCIOConfigArtifact(/cfg.ocio)"


class TestOCIOConfigArtifactDefaults:
    def test_default_context_vars_is_empty_dict(self) -> None:
        artifact = OCIOConfigArtifact(file_path="/cfg.ocio")
        assert artifact.context_vars == {}

    def test_context_vars_not_shared_between_instances(self) -> None:
        a = OCIOConfigArtifact(file_path="a")
        b = OCIOConfigArtifact(file_path="b")
        a.context_vars["X"] = "1"
        assert "X" not in b.context_vars
