import json
from pathlib import Path

import pytest

from intelligence_collector.config import DEFAULT_REPORT_GUIDELINES, LangfuseSettings, Settings, load_local_env


def test_local_env_does_not_override_process_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SOME_SETTING=from-file\n", encoding="utf-8")
    monkeypatch.setenv("SOME_SETTING", "from-process")

    load_local_env()

    assert __import__("os").environ["SOME_SETTING"] == "from-process"


def test_env_example_contains_placeholders_not_credentials():
    example = Path(".env.example").read_text(encoding="utf-8")
    assert "GOOGLE_API_KEY=" in example
    assert "GROQ_API_KEY=" in example
    assert "INTELLIGENCE_REASONING_MODEL=" in example
    assert "INTELLIGENCE_REASONING_PROVIDER=" in example
    assert "INTELLIGENCE_STANDARD_MODEL=" in example
    assert "RUN_LIVE_LLM=" in example
    assert "sk-" not in example


def test_langfuse_settings_use_base_url_and_trim_values(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "  public-key  ")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "  secret-key  ")
    monkeypatch.setenv("LANGFUSE_BASE_URL", " https://us.cloud.langfuse.com/ ")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    settings = LangfuseSettings.from_env(required=True)

    assert settings.public_key == "public-key"
    assert settings.secret_key == "secret-key"
    assert settings.host == "https://us.cloud.langfuse.com"


def test_report_guidelines_default_to_the_built_in_checklist(monkeypatch):
    monkeypatch.delenv("INTELLIGENCE_REPORT_GUIDELINES", raising=False)

    settings = Settings.from_env()

    assert settings.report_guidelines == DEFAULT_REPORT_GUIDELINES


def test_report_guidelines_are_overridable_via_env_json_array(monkeypatch):
    custom = ["The report MUST cite at least three independent sources."]
    monkeypatch.setenv("INTELLIGENCE_REPORT_GUIDELINES", json.dumps(custom))

    settings = Settings.from_env()

    assert settings.report_guidelines == custom


def test_report_guidelines_rejects_non_string_array(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_REPORT_GUIDELINES", json.dumps({"not": "a list"}))

    with pytest.raises(ValueError, match="INTELLIGENCE_REPORT_GUIDELINES"):
        Settings.from_env()


def test_eval_metrics_default_to_all_and_accept_a_subset(monkeypatch):
    import pytest

    from intelligence_collector.config import EVAL_METRICS

    monkeypatch.delenv("INTELLIGENCE_EVAL_METRICS", raising=False)
    assert Settings.from_env().eval_metrics == EVAL_METRICS

    monkeypatch.setenv("INTELLIGENCE_EVAL_METRICS", " Faithfulness , context_precision ")
    assert Settings.from_env().eval_metrics == ("faithfulness", "context_precision")

    monkeypatch.setenv("INTELLIGENCE_EVAL_METRICS", "")
    assert Settings.from_env().eval_metrics == ()

    monkeypatch.setenv("INTELLIGENCE_EVAL_METRICS", "faithfulness,bleu")
    with pytest.raises(ValueError, match="bleu"):
        Settings.from_env()
