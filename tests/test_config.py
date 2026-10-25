from pathlib import Path

import pytest

from backend.config import ConfigurationError, Settings


def test_loads_default_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_TIMEOUT_SECONDS",
        "MAX_DOCUMENT_SIZE_MB",
        "DATABASE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "qwen2.5:3b"
    assert settings.max_document_size_bytes == 5 * 1024 * 1024
    assert settings.database_path == Path("./data/sloth_ai.db")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("OLLAMA_BASE_URL", "localhost:11434", "OLLAMA_BASE_URL"),
        ("OLLAMA_BASE_URL", "http://", "OLLAMA_BASE_URL"),
        ("OLLAMA_BASE_URL", "http://[::1", "OLLAMA_BASE_URL"),
        ("OLLAMA_BASE_URL", "http://localhost:invalid", "OLLAMA_BASE_URL"),
        ("OLLAMA_BASE_URL", "http://user:password@localhost:11434", "OLLAMA_BASE_URL"),
        ("OLLAMA_MODEL", " ", "OLLAMA_MODEL"),
        ("OLLAMA_TIMEOUT_SECONDS", "0", "OLLAMA_TIMEOUT_SECONDS"),
        ("OLLAMA_TIMEOUT_SECONDS", "nan", "OLLAMA_TIMEOUT_SECONDS"),
        ("MAX_DOCUMENT_SIZE_MB", "-1", "MAX_DOCUMENT_SIZE_MB"),
        ("DATABASE_PATH", " ", "DATABASE_PATH"),
    ],
)
def test_invalid_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=message):
        Settings.from_env()
