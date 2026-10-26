import logging
from typing import Any

import requests

from backend.config import Settings

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Base error for local Ollama communication failures."""


class OllamaUnavailableError(OllamaError):
    """Ollama could not be reached at the configured endpoint."""


class OllamaModelNotFoundError(OllamaError):
    """The configured model is not installed in Ollama."""


class OllamaResponseError(OllamaError):
    """Ollama returned a response that did not match its API contract."""


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")

    def installed_models(self) -> set[str]:
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=min(self.settings.ollama_timeout_seconds, 5),
            )
            response.raise_for_status()
            payload: Any = response.json()
        except requests.Timeout as exc:
            raise OllamaUnavailableError(
                "Timed out while connecting to Ollama. Check that Ollama is running."
            ) from exc
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(
                "Could not connect to Ollama. Install or start Ollama, then try again."
            ) from exc
        except requests.HTTPError as exc:
            raise OllamaUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code} while listing models."
            ) from exc
        except requests.exceptions.JSONDecodeError as exc:
            raise OllamaResponseError(
                "Ollama returned invalid JSON while listing models."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                "Could not check Ollama availability."
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise OllamaResponseError(
                "Ollama returned an unexpected model-list response."
            )

        models: set[str] = set()
        for model in payload["models"]:
            if isinstance(model, dict):
                name = model.get("name") or model.get("model")
                if isinstance(name, str):
                    models.add(name)
        return models

    def ensure_model_available(self) -> None:
        installed = self.installed_models()
        if self.settings.ollama_model not in installed:
            raise OllamaModelNotFoundError(
                f"Model '{self.settings.ollama_model}' is not installed. "
                f"Run `ollama pull {self.settings.ollama_model}` or set OLLAMA_MODEL "
                "to an installed model."
            )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        logger.debug(
            "Ollama request started model=%s response_format=%s",
            self.settings.ollama_model,
            "json_schema" if json_schema is not None else "text",
        )
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
            "think": False,
        }
        if json_schema is not None:
            payload["format"] = json_schema

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.settings.ollama_timeout_seconds,
            )
            response.raise_for_status()
            result: Any = response.json()
        except requests.Timeout as exc:
            raise OllamaUnavailableError(
                "The Ollama request timed out. You can retry or increase "
                "OLLAMA_TIMEOUT_SECONDS."
            ) from exc
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(
                "Could not connect to Ollama. Check that it is running."
            ) from exc
        except requests.HTTPError as exc:
            status_code = exc.response.status_code
            if status_code == 404:
                raise OllamaModelNotFoundError(
                    f"Model '{self.settings.ollama_model}' is unavailable or not installed. "
                    f"Run `ollama pull {self.settings.ollama_model}`."
                ) from exc
            logger.warning(
                "Ollama request failed model=%s stage=http_response category=http_error "
                "status=%s",
                self.settings.ollama_model,
                status_code,
            )
            raise OllamaError(
                f"Ollama returned HTTP {status_code} while generating a response."
            ) from exc
        except requests.exceptions.JSONDecodeError as exc:
            logger.warning(
                "Ollama response rejected model=%s stage=http_json category=invalid_json",
                self.settings.ollama_model,
            )
            raise OllamaResponseError(
                "Ollama returned invalid JSON for the generation request."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                "The request to Ollama failed. Check its local service status."
            ) from exc

        if not isinstance(result, dict):
            logger.warning(
                "Ollama response rejected model=%s stage=response_envelope "
                "category=unexpected_shape",
                self.settings.ollama_model,
            )
            raise OllamaResponseError("Ollama returned an unexpected response.")
        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            logger.warning(
                "Ollama response rejected model=%s stage=response_content "
                "category=missing_or_empty",
                self.settings.ollama_model,
            )
            raise OllamaResponseError("Ollama returned an empty response.")
        return content.strip()


def ask_qwen(prompt: str) -> str:
    """Keep the original email-analysis API available to existing callers."""
    return OllamaClient(Settings.from_env()).generate(prompt)