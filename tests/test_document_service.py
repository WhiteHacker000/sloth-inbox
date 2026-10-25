from pathlib import Path

import pytest

from backend.document_service import (
    DocumentError,
    read_text_bytes,
    read_text_file,
    validate_text,
)


def test_accepts_valid_text_and_utf8_bom() -> None:
    assert validate_text("A document", 100) == "A document"
    assert read_text_bytes(b"\xef\xbb\xbfA document", 100) == "A document"


def test_rejects_empty_input() -> None:
    with pytest.raises(DocumentError, match="empty"):
        validate_text(" \n\t", 100)


def test_rejects_invalid_utf8() -> None:
    with pytest.raises(DocumentError, match="UTF-8"):
        read_text_bytes(b"\xff\xfe\x81", 100)


def test_rejects_oversized_text_and_upload() -> None:
    with pytest.raises(DocumentError, match="size limit"):
        validate_text("too long", 3)
    with pytest.raises(DocumentError, match="size limit"):
        read_text_bytes(b"too long", 3)


def test_reads_text_file(tmp_path: Path) -> None:
    document = tmp_path / "brief.txt"
    document.write_text("Project update", encoding="utf-8")
    assert read_text_file(document, 100) == "Project update"


def test_missing_file_and_unsupported_format_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DocumentError, match="does not exist"):
        read_text_file(tmp_path / "missing.txt", 100)

    pdf = tmp_path / "document.pdf"
    pdf.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(DocumentError, match="Only plain-text"):
        read_text_file(pdf, 100)
