import pytest
from pydantic import ValidationError

from backend.schemas import DocumentAnalysis


def test_valid_analysis_and_missing_optional_values() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "summary": "A short summary.",
            "tasks": [
                {
                    "description": "Send the draft",
                    "deadline_text": "next Friday",
                }
            ],
            "events": [
                {"description": "Review meeting", "date_text": "March 2"}
            ],
            "potential_blockers": [],
        }
    )

    assert analysis.tasks[0].owner is None
    assert analysis.tasks[0].priority is None
    assert analysis.tasks[0].status == "pending"
    assert analysis.tasks[0].deadline_text == "next Friday"
    assert analysis.events[0].date_text == "March 2"


def test_empty_lists_are_valid() -> None:
    result = DocumentAnalysis.model_validate(
        {"summary": "Nothing actionable is stated."}
    )
    assert result.tasks == []
    assert result.events == []
    assert result.potential_blockers == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"summary": 3},
        {"summary": "Summary", "tasks": ["not a task"]},
        {
            "summary": "Summary",
            "tasks": [{"description": "Task", "priority": "critical"}],
        },
        {
            "summary": "Summary",
            "events": [{"description": "Event", "date_text": 20260101}],
        },
        {"summary": "Summary", "potential_blockers": [""]},
        {"summary": "x" * 501},
        {"summary": "Summary", "unexpected": "field"},
    ],
)
def test_invalid_types_and_enum_values_are_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        DocumentAnalysis.model_validate(payload)


def test_ambiguous_date_is_preserved_without_normalization() -> None:
    result = DocumentAnalysis.model_validate(
        {
            "summary": "Review planned.",
            "events": [
                {
                    "description": "Review",
                    "date_text": "next Friday",
                }
            ],
        }
    )
    assert result.events[0].date_text == "next Friday"


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DocumentAnalysis.model_validate_json('{"summary": ')
