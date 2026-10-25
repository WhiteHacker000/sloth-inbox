from datetime import UTC, date, datetime
from pathlib import Path

from backend.briefing import build_daily_briefing, classify_deadline, parse_explicit_deadline
from backend.database import DocumentRepository
from backend.schemas import DocumentAnalysis


def test_deadline_parser_requires_complete_unambiguous_date() -> None:
    assert parse_explicit_deadline("2026-10-24") == date(2026, 10, 24)
    assert parse_explicit_deadline("October 27, 2026") == date(2026, 10, 27)
    assert parse_explicit_deadline("27 Oct 2026") == date(2026, 10, 27)
    assert parse_explicit_deadline("tomorrow") is None
    assert parse_explicit_deadline("Friday") is None
    assert parse_explicit_deadline("10/11/2026") is None
    assert parse_explicit_deadline("October 2026") is None


def test_briefing_counts_and_deadlines_come_from_database(
    tmp_path: Path,
) -> None:
    repository = DocumentRepository(tmp_path / "assistant.db")
    analysis = DocumentAnalysis.model_validate(
        {
            "summary": "A plan with due dates.",
            "tasks": [
                {
                    "description": "Overdue report",
                    "deadline_text": "2026-10-24",
                },
                {
                    "description": "Upcoming review",
                    "deadline_text": "October 27, 2026",
                },
                {
                    "description": "Ambiguous reminder",
                    "deadline_text": "Friday",
                },
                {
                    "description": "Completed old task",
                    "deadline_text": "2026-10-20",
                },
                {"description": "No deadline"},
            ],
            "potential_blockers": ["Approval is pending."],
        }
    )
    analysis_id = repository.save_analysis("Original plan", analysis, "Plan")
    tasks = repository.list_tasks()
    completed_task_id = next(
        task["id"] for task in tasks if task["description"] == "Completed old task"
    )
    repository.set_task_status(completed_task_id, "completed")

    briefing = build_daily_briefing(
        repository,
        today=date(2026, 10, 25),
        now=datetime(2026, 10, 25, 12, tzinfo=UTC),
    )

    assert briefing["counts"] == {
        "pending": 4,
        "in_progress": 0,
        "completed": 1,
    }
    assert [task["description"] for task in briefing["overdue_tasks"]] == [
        "Overdue report"
    ]
    assert [task["description"] for task in briefing["upcoming_deadlines"]] == [
        "Upcoming review"
    ]
    assert [task["description"] for task in briefing["unresolved_deadlines"]] == [
        "Ambiguous reminder"
    ]
    assert briefing["completed_tasks"][0]["description"] == "Completed old task"
    assert briefing["recent_analyses"][0]["id"] == analysis_id
    assert briefing["potential_blockers"][0]["description"] == "Approval is pending."
    assert briefing["activity"]["analyses"] == 1
    assert briefing["activity"]["tasks"] == 5
    assert briefing["activity"]["successful_runs"] == 0


def test_deadline_classification_keeps_original_text_and_no_deadline_separate() -> None:
    state, parsed = classify_deadline("next Friday", date(2026, 10, 25))
    assert state == "unresolved"
    assert parsed is None
    assert classify_deadline(None, date(2026, 10, 25)) == ("none", None)
