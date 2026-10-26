import json
from io import BytesIO
from pathlib import Path
from typing import Callable

import pytest
from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from streamlit.testing.v1 import AppTest

from backend.ai import OllamaClient, OllamaModelNotFoundError, OllamaUnavailableError
from backend.database import DocumentRepository


def test_local_model_card_hides_endpoint_and_model_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "status.db"))
    monkeypatch.setattr(OllamaClient, "ensure_model_available", lambda self: None)

    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(
        timeout=60
    )

    assert not app.exception
    assert "Local Model" in [header.value for header in app.header]
    assert any(
        success.value == "Ollama is available and the configured model is installed."
        for success in app.success
    )
    assert all("localhost:11434" not in caption.value for caption in app.caption)
    assert all("qwen2.5:3b" not in caption.value for caption in app.caption)


def test_gemma_sidebar_selection_is_used_for_document_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "gemma.db"))
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    monkeypatch.setattr(OllamaClient, "ensure_model_available", lambda self: None)
    model_calls: list[str] = []
    response = json.dumps(
        {
            "summary": "Configure DNS.",
            "tasks": [
                {
                    "description": "Configure DNS",
                    "evidence": "Configure DNS.",
                }
            ],
            "events": [],
            "potential_blockers": [],
        }
    )

    def generate(
        self: OllamaClient,
        prompt: str,
        **kwargs: object,
    ) -> str:
        model_calls.append(self.settings.ollama_model)
        return response

    monkeypatch.setattr(OllamaClient, "generate", generate)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(
        timeout=60
    )

    model_selectbox = next(
        widget for widget in app.selectbox if widget.label == "Ollama model"
    )
    model_selectbox.select("gemma3:1b").run(timeout=60)
    app.text_area[0].set_value("Configure DNS.").run()
    analyze_button = next(
        button for button in app.button if button.label == "Analyze document"
    )
    app = analyze_button.click().run(timeout=60)

    assert not app.exception
    assert model_calls == ["gemma3:1b"]
    assert any(
        success.value == "Analysis completed and saved locally."
        for success in app.success
    )


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (
            OllamaUnavailableError("Could not connect to http://localhost:11434"),
            "Ollama is unavailable. Start the local service and try again.",
        ),
        (
            OllamaModelNotFoundError("Model qwen2.5:3b is not installed."),
            "Ollama is running, but the configured model is unavailable.",
        ),
    ],
)
def test_local_model_card_shows_safe_failure_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: OllamaUnavailableError | OllamaModelNotFoundError,
    expected_message: str,
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "status.db"))

    def raise_error(self: OllamaClient) -> None:
        raise error

    monkeypatch.setattr(OllamaClient, "ensure_model_available", raise_error)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(
        timeout=60
    )

    assert not app.exception
    assert any(expected_message in warning.value for warning in app.warning)
    assert all("localhost:11434" not in warning.value for warning in app.warning)
    assert all("qwen2.5:3b" not in warning.value for warning in app.warning)


@pytest.mark.parametrize(
    "source_text",
    [
        "1. Configure DNS on the client machines.\n2. Verify failover.",
        "The project was founded in 2018 and operates in three regions.",
    ],
)
def test_empty_extraction_shows_review_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_text: str,
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "review.db"))
    monkeypatch.setattr(OllamaClient, "ensure_model_available", lambda self: None)
    monkeypatch.setattr(
        OllamaClient,
        "generate",
        lambda self, prompt, **kwargs: json.dumps(
            {
                "summary": "Technical implementation requirements.",
                "tasks": [],
                "events": [],
                "potential_blockers": [],
            }
        ),
    )

    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(
        timeout=60
    )
    app.text_area[0].set_value(source_text).run()
    analyze_button = next(
        button for button in app.button if button.label == "Analyze document"
    )
    app = analyze_button.click().run(timeout=60)

    assert not app.exception
    assert any(
        "The document may contain no actionable content" in warning.value
        for warning in app.warning
    )
    assert all(
        success.value != "Analysis completed and saved locally."
        for success in app.success
    )


def _make_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 72 240 Td (" + text.encode("ascii") + b") Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _make_docx(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    output = BytesIO()
    document.save(output)
    return output.getvalue()


@pytest.mark.parametrize(
    ("filename", "mime_type", "make_content"),
    [
        ("brief.txt", "text/plain", lambda text: text.encode("utf-8")),
        ("brief.pdf", "application/pdf", _make_pdf),
        (
            "brief.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            _make_docx,
        ),
    ],
)
def test_uploaded_documents_are_analyzed_persisted_and_tasks_remain_manageable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    mime_type: str,
    make_content: Callable[[str], bytes],
) -> None:
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "uploads.db"))
    monkeypatch.setattr(OllamaClient, "ensure_model_available", lambda self: None)
    monkeypatch.setattr(
        OllamaClient,
        "generate",
        lambda self, prompt, **kwargs: json.dumps(
            {
                "summary": "Configure the DNS resolver.",
                "tasks": [
                    {
                        "description": "Configure the DNS resolver",
                        "evidence": "Configure the DNS resolver.",
                    }
                ],
                "events": [],
                "potential_blockers": [],
            }
        ),
    )

    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(
        timeout=60
    )
    app.file_uploader[0].set_value(
        (filename, make_content("Configure the DNS resolver."), mime_type)
    ).run()
    analyze_button = next(
        button for button in app.button if button.label == "Analyze document"
    )
    app = analyze_button.click().run(timeout=60)

    assert not app.exception
    assert any(
        success.value == "Analysis completed and saved locally."
        for success in app.success
    )
    assert any(
        "Configure the DNS resolver." in str(element.value)
        for element in app.markdown
    )
    repository = DocumentRepository(tmp_path / "uploads.db")
    analyses = repository.list_analyses()
    assert len(analyses) == 1
    assert analyses[0]["title"] == filename
    saved = repository.get_analysis(analyses[0]["id"])
    assert saved["source_text"] == "Configure the DNS resolver."
    assert saved["analysis"].tasks[0].evidence == "Configure the DNS resolver."

    complete_button = next(
        button for button in app.button if button.label == "Mark completed"
    )
    app = complete_button.click().run(timeout=60)
    assert not app.exception
    assert repository.list_tasks()[0]["status"] == "completed"
    assert any(selectbox.label == "Select an analysis" for selectbox in app.selectbox)
    assert "Daily productivity briefing" in [header.value for header in app.subheader]

    reopen_button = next(
        button for button in app.button if button.label == "Reopen task"
    )
    app = reopen_button.click().run(timeout=60)
    assert not app.exception
    assert repository.list_tasks()[0]["status"] == "pending"
