from pathlib import Path


class DocumentError(ValueError):
    """Raised when a document cannot be safely accepted for analysis."""


def validate_text(text: str, max_size_bytes: int) -> str:
    if not isinstance(text, str):
        raise DocumentError("Document content must be UTF-8 text.")
    if not text.strip():
        raise DocumentError("The document is empty. Add text before analyzing.")
    try:
        size_bytes = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise DocumentError("The document contains text that cannot be encoded as UTF-8.") from exc
    if size_bytes > max_size_bytes:
        max_size_mb = max_size_bytes / (1024 * 1024)
        raise DocumentError(
            f"The document exceeds the configured size limit of {max_size_mb:g} MB."
        )
    return text


def read_text_file(path: str | Path, max_size_bytes: int) -> str:
    file_path = Path(path)
    if file_path.suffix.lower() != ".txt":
        raise DocumentError("Only plain-text .txt files are supported.")
    if not file_path.is_file():
        raise DocumentError("The selected file does not exist or is not a regular file.")
    try:
        if file_path.stat().st_size > max_size_bytes:
            raise DocumentError("The document exceeds the configured size limit.")
        content = file_path.read_bytes()
    except OSError as exc:
        raise DocumentError("The selected file could not be read.") from exc
    return read_text_bytes(content, max_size_bytes)


def read_text_bytes(content: bytes, max_size_bytes: int) -> str:
    if len(content) > max_size_bytes:
        raise DocumentError("The document exceeds the configured size limit.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentError("The file is not valid UTF-8 text.") from exc
    return validate_text(text, max_size_bytes)
