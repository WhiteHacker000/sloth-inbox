from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, is_zipfile

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader


class DocumentError(ValueError):
    """Raised when a document cannot be safely accepted for analysis."""


SUPPORTED_DOCUMENT_EXTENSIONS = (".txt", ".pdf", ".docx")


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


def extract_document_text(
    content: bytes,
    filename: str,
    max_size_bytes: int,
) -> str:
    """Extract validated text from a supported upload without writing it to disk."""
    if len(content) > max_size_bytes:
        raise DocumentError("The document exceeds the configured size limit.")
    if not content:
        raise DocumentError("The document is empty.")

    extension = Path(filename).suffix.lower()
    if extension == ".txt":
        return read_text_bytes(content, max_size_bytes)
    if extension == ".pdf":
        return _extract_pdf_text(content, max_size_bytes)
    if extension == ".docx":
        return _extract_docx_text(content, max_size_bytes)
    supported = ", ".join(SUPPORTED_DOCUMENT_EXTENSIONS)
    raise DocumentError(f"Unsupported document type. Supported formats: {supported}.")


def _extract_pdf_text(content: bytes, max_size_bytes: int) -> str:
    if content[:1024].find(b"%PDF-") < 0:
        raise DocumentError("The uploaded file does not have a valid PDF signature.")
    try:
        reader = PdfReader(BytesIO(content), strict=False)
        if reader.is_encrypted:
            raise DocumentError("Password-protected PDF files are not supported.")
        pages = [page.extract_text() or "" for page in reader.pages]
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("The PDF is malformed or could not be read.") from exc

    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text.strip():
        raise DocumentError(
            "No selectable text was found in this PDF. Scanned PDFs requiring OCR "
            "are not supported yet."
        )
    return validate_text(text, max_size_bytes)


def _extract_docx_text(content: bytes, max_size_bytes: int) -> str:
    try:
        if not is_zipfile(BytesIO(content)):
            raise DocumentError("The uploaded file is not a valid DOCX document.")
        with BytesIO(content) as stream:
            document = Document(stream)
        sections: list[str] = []
        body = document.element.body
        for child in body.iterchildren():
            if child.tag.endswith("}p"):
                paragraph = Paragraph(child, document)
                if paragraph.text.strip():
                    sections.append(paragraph.text)
            elif child.tag.endswith("}tbl"):
                table = Table(child, document)
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    row_text = " | ".join(cell for cell in cells if cell)
                    if row_text:
                        sections.append(row_text)
        text = "\n".join(sections)
    except DocumentError:
        raise
    except (BadZipFile, KeyError, ValueError, TypeError, OSError) as exc:
        raise DocumentError("The DOCX file is malformed or could not be read.") from exc
    except Exception as exc:
        raise DocumentError("The DOCX file is malformed or could not be read.") from exc
    return validate_text(text, max_size_bytes)
