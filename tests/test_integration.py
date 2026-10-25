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
        '{"summary":"The MVP is nearing its deadline.",'
        '"tasks":['
        '{"description":"Submit the MVP","owner":"development team",'
        '"deadline_text":"October 27, 2026",'
        '"evidence":"The development team must submit the MVP by October 27, 2026."},'
        '{"description":"Complete the authentication module","owner":"Rahul",'
        '"deadline_text":"October 26, 2026",'
        '"evidence":"Rahul must complete the authentication module by October 26, 2026."},'
        '{"description":"Review the dashboard design","owner":"UI team",'
        '"evidence":"The UI team must review the dashboard design."},'
        '{"description":"Resolve the database connection issue",'
        '"deadline_text":"before deployment",'
        '"evidence":"We need to resolve the database connection issue before deployment."},'
        '{"description":"Update the final README with installation and execution '
        'instructions",'
        '"evidence":"The final README must include installation and execution instructions."}'
        '],"events":['
        '{"description":"Project demonstration",'
        '"date_text":"October 28, 2026, at 3 PM",'
        '"evidence":"The project demonstration is scheduled for October 28, 2026, at 3 PM."}'
        '],"potential_blockers":['
        '"Deployment is blocked until the production database configuration is verified."'
        ']}'
    )
    analysis = analyze_document(
        "The development team must submit the MVP by October 27, 2026. "
        "Rahul must complete the authentication module by October 26, 2026. "
        "The UI team must review the dashboard design. "
        "We need to resolve the database connection issue before deployment. "
        "The project demonstration is scheduled for October 28, 2026, at 3 PM. "
        "The final README must include installation and execution instructions. "
        "Deployment is blocked until the production database configuration is verified.",
        client,
        "MVP plan",
    )
    repository = DocumentRepository(tmp_path / "sloth.db")
    analysis_id = repository.save_analysis(
        "The development team must submit the MVP by October 27, 2026. "
        "Rahul must complete the authentication module by October 26, 2026. "
        "The UI team must review the dashboard design. "
        "We need to resolve the database connection issue before deployment. "
        "The project demonstration is scheduled for October 28, 2026, at 3 PM. "
        "The final README must include installation and execution instructions. "
        "Deployment is blocked until the production database configuration is verified.",
        analysis,
        "MVP plan",
    )

    restored = repository.get_analysis(analysis_id)
    assert isinstance(restored["analysis"], DocumentAnalysis)
    saved_analysis = restored["analysis"]
    assert len(saved_analysis.tasks) == 5
    assert saved_analysis.tasks[0].deadline_text == "October 27, 2026"
    assert saved_analysis.tasks[0].evidence == (
        "The development team must submit the MVP by October 27, 2026."
    )
    assert len(saved_analysis.events) == 1
    assert saved_analysis.events[0].date_text == "October 28, 2026, at 3 PM"
    assert saved_analysis.events[0].evidence == (
        "The project demonstration is scheduled for October 28, 2026, at 3 PM."
    )
    assert saved_analysis.potential_blockers == [
        "Deployment is blocked until the production database configuration is verified."
    ]
    saved_tasks = repository.list_tasks()
    assert len(saved_tasks) == 5
    submitted_mvp = next(
        task for task in saved_tasks if task["description"] == "Submit the MVP"
    )
    assert submitted_mvp["deadline_text"] == "October 27, 2026"
    assert submitted_mvp["evidence"] == (
        "The development team must submit the MVP by October 27, 2026."
    )


def test_invalid_ai_output_does_not_create_database_record(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "sloth.db")

    with pytest.raises(AnalysisOutputError):
        analyze_document("Document", MockOllama("not JSON"))

    assert repository.list_analyses() == []
    assert repository.list_tasks() == []


def test_technical_evaluation_preserves_tasks_event_and_genuine_blocker(
    tmp_path: Path,
) -> None:
    source = (
        "1. Create a topology and IP/service inventory.\n"
        "2. Configure the DNS resolver on all client machines.\n"
        "3. Configure the backup DNS resolver.\n"
        "4. Ensure backend ports accept connections only from the nginx reverse proxy.\n"
        "5. Run the fault injection challenge and verify recovery.\n"
        "6. Submit the final report with implementation changes and test results by "
        "October 27, 2026.\n"
        "The project demonstration is scheduled for October 28 at 3 PM.\n"
        "Deployment cannot proceed because the production database is unreachable."
    )
    response = (
        '{"summary":"Technical project evaluation deliverables and deployment issue.",'
        '"tasks":['
        '{"description":"Create a topology and IP/service inventory",'
        '"evidence":"1. Create a topology and IP/service inventory."},'
        '{"description":"Configure the DNS resolver on all client machines",'
        '"evidence":"2. Configure the DNS resolver on all client machines."},'
        '{"description":"Configure the backup DNS resolver",'
        '"evidence":"3. Configure the backup DNS resolver."},'
        '{"description":"Ensure backend ports accept connections only from the nginx reverse proxy",'
        '"evidence":"4. Ensure backend ports accept connections only from the nginx reverse proxy."},'
        '{"description":"Run the fault injection challenge and verify recovery",'
        '"evidence":"5. Run the fault injection challenge and verify recovery."},'
        '{"description":"Submit the final report with implementation changes and test results",'
        '"deadline_text":"October 27, 2026",'
        '"evidence":"6. Submit the final report with implementation changes and test results by October 27, 2026."}'
        '],"events":['
        '{"description":"Project demonstration",'
        '"date_text":"October 28 at 3 PM",'
        '"evidence":"The project demonstration is scheduled for October 28 at 3 PM."}'
        '],"potential_blockers":['
        '"Deployment cannot proceed because the production database is unreachable."'
        ']}'
    )

    analysis = analyze_document(source, MockOllama(response), "Technical evaluation")
    repository = DocumentRepository(tmp_path / "evaluation.db")
    analysis_id = repository.save_analysis(source, analysis, "Technical evaluation")

    saved = repository.get_analysis(analysis_id)["analysis"]
    assert isinstance(saved, DocumentAnalysis)
    assert len(saved.tasks) == 6
    assert [task.evidence for task in saved.tasks] == [
        "1. Create a topology and IP/service inventory.",
        "2. Configure the DNS resolver on all client machines.",
        "3. Configure the backup DNS resolver.",
        "4. Ensure backend ports accept connections only from the nginx reverse proxy.",
        "5. Run the fault injection challenge and verify recovery.",
        "6. Submit the final report with implementation changes and test results by October 27, 2026.",
    ]
    final_report = saved.tasks[-1]
    assert final_report.deadline_text == "October 27, 2026"
    assert len(saved.events) == 1
    assert saved.events[0].date_text == "October 28 at 3 PM"
    assert saved.events[0].evidence == (
        "The project demonstration is scheduled for October 28 at 3 PM."
    )
    assert saved.potential_blockers == [
        "Deployment cannot proceed because the production database is unreachable."
    ]
    persisted_tasks = repository.list_tasks_for_analysis(analysis_id)
    assert len(persisted_tasks) == 6
    persisted_report = next(
        task for task in persisted_tasks if task["description"].startswith("Submit the final report")
    )
    assert persisted_report["deadline_text"] == "October 27, 2026"
