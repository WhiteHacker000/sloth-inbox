import logging
import re
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal

from dateutil import parser as date_parser
from pydantic import ValidationError

from backend.database import DocumentRepository
from backend.schemas import DocumentAnalysis

logger = logging.getLogger(__name__)

DeadlineState = Literal["upcoming", "overdue", "unresolved", "none"]

_ISO_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_WRITTEN_DATE = re.compile(
    rf"\b(?:{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+\d{{4}}|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}[,]?\s+\d{{4}})\b",
    re.IGNORECASE,
)


def parse_explicit_deadline(deadline_text: str | None) -> date | None:
    """Parse only full, unambiguous dates; never supply a missing year."""
    if not deadline_text:
        return None
    iso_match = _ISO_DATE.search(deadline_text)
    if iso_match:
        try:
            return date.fromisoformat(iso_match.group(1))
        except ValueError:
            return None
    written_match = _WRITTEN_DATE.search(deadline_text)
    if written_match:
        try:
            parsed = date_parser.parse(
                written_match.group(0),
                fuzzy=False,
                default=datetime(2000, 1, 1),
            )
            return parsed.date()
        except (ValueError, OverflowError):
            return None
    return None


def classify_deadline(
    deadline_text: str | None,
    today: date,
) -> tuple[DeadlineState, date | None]:
    if not deadline_text:
        return "none", None
    resolved = parse_explicit_deadline(deadline_text)
    if resolved is None:
        return "unresolved", None
    return ("overdue" if resolved < today else "upcoming"), resolved


def build_daily_briefing(
    repository: DocumentRepository,
    *,
    today: date | None = None,
    now: datetime | None = None,
    recent_days: int = 7,
) -> dict[str, object]:
    current_time = now or datetime.now(UTC)
    current_date = today or current_time.date()
    start_of_today = datetime.combine(current_date, time.min, tzinfo=UTC)
    recent_since = start_of_today - timedelta(days=recent_days)
    data = repository.get_briefing_data(recent_since, start_of_today)

    tasks = data["tasks"]
    if not isinstance(tasks, list):
        raise TypeError("Repository returned invalid briefing tasks.")

    pending: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    upcoming: list[dict[str, object]] = []
    overdue: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for task_value in tasks:
        if not isinstance(task_value, dict):
            raise TypeError("Repository returned an invalid task record.")
        task = dict(task_value)
        status = task["status"]
        if status == "completed":
            completed.append(task)
        else:
            pending.append(task)
        deadline_text = task.get("deadline_text")
        deadline_state, parsed_deadline = classify_deadline(
            deadline_text if isinstance(deadline_text, str) else None,
            current_date,
        )
        task["deadline_state"] = deadline_state
        task["deadline_date"] = parsed_deadline.isoformat() if parsed_deadline else None
        if status != "completed":
            if deadline_state == "upcoming":
                upcoming.append(task)
            elif deadline_state == "overdue":
                overdue.append(task)
            elif deadline_state == "unresolved":
                unresolved.append(task)

    recent_analyses: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    for analysis_row in data["recent_analyses"]:
        if not isinstance(analysis_row, dict):
            continue
        try:
            analysis = DocumentAnalysis.model_validate_json(
                analysis_row["analysis_json"]
            )
        except (ValidationError, ValueError, TypeError):
            logger.warning(
                "Skipping invalid saved analysis id=%s",
                analysis_row.get("id"),
            )
            continue
        recent_analyses.append(
            {
                "id": analysis_row["id"],
                "title": analysis_row["title"],
                "created_at": analysis_row["created_at"],
                "summary": analysis.summary,
            }
        )
        blockers.extend(
            {
                "analysis_id": analysis_row["id"],
                "title": analysis_row["title"],
                "created_at": analysis_row["created_at"],
                "description": blocker,
            }
            for blocker in analysis.potential_blockers
        )

    return {
        "today": current_date.isoformat(),
        "counts": data["counts"],
        "pending_tasks": pending,
        "completed_tasks": completed,
        "upcoming_deadlines": upcoming,
        "overdue_tasks": overdue,
        "unresolved_deadlines": unresolved,
        "recent_analyses": recent_analyses,
        "potential_blockers": blockers,
        "activity": data["totals"],
    }