import os
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when an environment setting is invalid."""


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: float
    max_document_size_mb: int
    database_path: Path
    document_input_dir: Path | None = None
    document_poll_interval_seconds: float = 30
    document_stability_seconds: float = 2
    document_recursive: bool = False
    document_max_attempts: int = 3

    @property
    def max_document_size_bytes(self) -> int:
        return self.max_document_size_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
        timeout_value = os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")
        size_value = os.getenv("MAX_DOCUMENT_SIZE_MB", "5")
        database_path = os.getenv("DATABASE_PATH", "./data/sloth_ai.db").strip()
        document_input_dir = os.getenv("DOCUMENT_INPUT_DIR", "").strip()
        poll_interval_value = os.getenv("DOCUMENT_POLL_INTERVAL_SECONDS", "30")
        stability_value = os.getenv("DOCUMENT_STABILITY_SECONDS", "2")
        recursive_value = os.getenv("DOCUMENT_RECURSIVE", "false").strip().lower()
        max_attempts_value = os.getenv("DOCUMENT_MAX_ATTEMPTS", "3")

        try:
            parsed_url = urlsplit(base_url)
            hostname = parsed_url.hostname
            port = parsed_url.port
        except ValueError as exc:
            raise ConfigurationError(
                "OLLAMA_BASE_URL must be a valid http:// or https:// URL."
            ) from exc
        if (
            parsed_url.scheme not in {"http", "https"}
            or not hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ConfigurationError(
                "OLLAMA_BASE_URL must be a valid http:// or https:// URL without credentials."
            )
        if not model:
            raise ConfigurationError("OLLAMA_MODEL must not be empty.")
        if not database_path:
            raise ConfigurationError("DATABASE_PATH must not be empty.")

        try:
            timeout = float(timeout_value)
        except ValueError as exc:
            raise ConfigurationError(
                "OLLAMA_TIMEOUT_SECONDS must be a positive number."
            ) from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ConfigurationError(
                "OLLAMA_TIMEOUT_SECONDS must be a positive number."
            )

        try:
            max_size = int(size_value)
        except ValueError as exc:
            raise ConfigurationError(
                "MAX_DOCUMENT_SIZE_MB must be a positive whole number."
            ) from exc
        if max_size <= 0:
            raise ConfigurationError(
                "MAX_DOCUMENT_SIZE_MB must be a positive whole number."
            )

        try:
            poll_interval = float(poll_interval_value)
            stability_seconds = float(stability_value)
        except ValueError as exc:
            raise ConfigurationError(
                "Document polling and stability intervals must be positive numbers."
            ) from exc
        if (
            not math.isfinite(poll_interval)
            or poll_interval <= 0
            or not math.isfinite(stability_seconds)
            or stability_seconds < 0
        ):
            raise ConfigurationError(
                "Document polling interval must be positive and stability seconds "
                "must be zero or greater."
            )
        if recursive_value not in {"true", "false"}:
            raise ConfigurationError("DOCUMENT_RECURSIVE must be true or false.")
        try:
            max_attempts = int(max_attempts_value)
        except ValueError as exc:
            raise ConfigurationError(
                "DOCUMENT_MAX_ATTEMPTS must be a positive whole number."
            ) from exc
        if max_attempts <= 0:
            raise ConfigurationError(
                "DOCUMENT_MAX_ATTEMPTS must be a positive whole number."
            )

        return cls(
            ollama_base_url=base_url,
            ollama_model=model,
            ollama_timeout_seconds=timeout,
            max_document_size_mb=max_size,
            database_path=Path(database_path).expanduser(),
            document_input_dir=(
                Path(document_input_dir).expanduser() if document_input_dir else None
            ),
            document_poll_interval_seconds=poll_interval,
            document_stability_seconds=stability_seconds,
            document_recursive=recursive_value == "true",
            document_max_attempts=max_attempts,
        )
