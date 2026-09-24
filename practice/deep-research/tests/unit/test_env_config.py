from pathlib import Path

from intelligence_collector.config import LangfuseSettings, load_local_env


def test_local_env_does_not_override_process_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SCOPE_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("SCOPE_MODEL", "from-process")

    load_local_env()

    assert __import__("os").environ["SCOPE_MODEL"] == "from-process"


def test_env_example_contains_placeholders_not_credentials():
    example = Path(".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in example
    assert "SCOPE_MODEL=" in example
    assert "SCOPE_PROVIDER=" in example
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
