from pathlib import Path

import pytest
import requests
from requests.exceptions import JSONDecodeError

from backend.ai import (
    OllamaClient,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaUnavailableError,
)
from backend.config import Settings


def make_client() -> OllamaClient:
    return OllamaClient(
        Settings(
            ollama_base_url="http://localhost:11434/",
            ollama_model="qwen2.5:3b",
            ollama_timeout_seconds=12,
            max_document_size_mb=5,
            database_path=Path(":memory:"),
        )
    )


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> object:
        return self.payload


def test_generates_text_and_sends_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def post(url: str, **kwargs: object) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse({"message": {"content": "  {\"ok\":true}  "}})

    monkeypatch.setattr("backend.ai.requests.post", post)
    result = make_client().generate("prompt", json_schema={"type": "object"})

    assert result == '{"ok":true}'
    assert calls[0]["url"] == "http://localhost:11434/api/chat"
    assert calls[0]["json"]["format"] == {"type": "object"}
    assert calls[0]["json"]["think"] is False
    assert "tools" not in calls[0]["json"]
    assert calls[0]["json"]["messages"][0]["content"] == "prompt"
    assert calls[0]["timeout"] == 12


def test_detects_available_and_missing_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backend.ai.requests.get",
        lambda *args, **kwargs: FakeResponse(
            {"models": [{"name": "qwen2.5:3b"}, {"model": "other:latest"}]}
        ),
    )
    make_client().ensure_model_available()

    monkeypatch.setattr(
        "backend.ai.requests.get",
        lambda *args, **kwargs: FakeResponse({"models": [{"name": "other:latest"}]}),
    )
    with pytest.raises(OllamaModelNotFoundError, match="ollama pull qwen2.5:3b"):
        make_client().ensure_model_available()


def test_connection_failure_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def get(*args: object, **kwargs: object) -> None:
        raise requests.ConnectionError

    monkeypatch.setattr("backend.ai.requests.get", get)
    with pytest.raises(OllamaUnavailableError, match="Install or start Ollama"):
        make_client().installed_models()


def test_timeout_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def post(*args: object, **kwargs: object) -> None:
        raise requests.Timeout

    monkeypatch.setattr("backend.ai.requests.post", post)
    with pytest.raises(OllamaUnavailableError, match="timed out"):
        make_client().generate("prompt")


def test_http_error_and_missing_model_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backend.ai.requests.post",
        lambda *args, **kwargs: FakeResponse({}, status_code=500),
    )
    with pytest.raises(OllamaError, match="HTTP 500"):
        make_client().generate("prompt")

    monkeypatch.setattr(
        "backend.ai.requests.post",
        lambda *args, **kwargs: FakeResponse({}, status_code=404),
    )
    with pytest.raises(OllamaModelNotFoundError, match="not installed"):
        make_client().generate("prompt")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": {"content": ""}},
        [],
    ],
)
def test_malformed_api_responses_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    monkeypatch.setattr(
        "backend.ai.requests.post",
        lambda *args, **kwargs: FakeResponse(payload),
    )
    with pytest.raises(OllamaResponseError):
        make_client().generate("prompt")


def test_invalid_model_list_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backend.ai.requests.get",
        lambda *args, **kwargs: FakeResponse({"not_models": []}),
    )
    with pytest.raises(OllamaResponseError, match="model-list"):
        make_client().installed_models()


def test_invalid_json_from_ollama_is_a_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_json() -> None:
        raise JSONDecodeError("invalid JSON", "not json", 0)

    monkeypatch.setattr(
        "backend.ai.requests.post",
        lambda *args, **kwargs: type(
            "InvalidJsonResponse",
            (),
            {"raise_for_status": lambda self: None, "json": lambda self: invalid_json()},
        )(),
    )

    with pytest.raises(OllamaResponseError, match="invalid JSON"):
        make_client().generate("prompt")

    monkeypatch.setattr(
        "backend.ai.requests.get",
        lambda *args, **kwargs: type(
            "InvalidJsonResponse",
            (),
            {"raise_for_status": lambda self: None, "json": lambda self: invalid_json()},
        )(),
    )
    with pytest.raises(OllamaResponseError, match="invalid JSON"):
        make_client().installed_models()
