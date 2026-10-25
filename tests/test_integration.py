from typing import Any
from pathlib import Path

import pytest

from backend.analyzer import AnalysisOutputError, analyze_document
from backend.database import DocumentRepository
from backend.schemas import DocumentAnalysis


class MockOllama:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        return self.response


def test_document_to_mocked_ai_to_sqlite_round_trip(tmp_path: Path) -> None:
    client = MockOllama(
        '{"summary":"Plan the launch.",'
        '"tasks":[{"description":"Send invitations","deadline_text":"Friday"}],'
        '"events":[{"description":"Launch review","date_text":"next Tuesday"}],'
        '"potential_blockers":["Design approval is pending."]}'
    )
    analysis = analyze_document(
        "Send invitations by Friday. Launch review next Tuesday.",
        client,
        "Launch plan",
    )
    repository = DocumentRepository(tmp_path / "sloth.db")
    analysis_id = repository.save_analysis(
        "Send invitations by Friday. Launch review next Tuesday.",
        analysis,
        "Launch plan",
    )

    restored = repository.get_analysis(analysis_id)
    assert isinstance(restored["analysis"], DocumentAnalysis)
    assert restored["analysis"].events[0].date_text == "next Tuesday"
    assert repository.list_tasks()[0]["deadline_text"] == "Friday"


def test_invalid_ai_output_does_not_create_database_record(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "sloth.db")

    with pytest.raises(AnalysisOutputError):
        analyze_document("Document", MockOllama("not JSON"))

    assert repository.list_analyses() == []
    assert repository.list_tasks() == []
