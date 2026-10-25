import json
from pathlib import Path
from typing import Any

import pytest

from backend.analyzer import should_review_empty_extraction
from backend.schemas import DocumentAnalysis


FIXTURE_DIR = Path(__file__).parent / "fixtures"
GROUND_TRUTH_PATH = FIXTURE_DIR / "cn_project_doc_ground_truth.json"
ARCHIVED_OUTPUT_PATH = FIXTURE_DIR / "cn_project_doc_archived_output.json"

ARCHIVED_MATCHES = (
    "phase1_task_a_private_lan",
    "phase1_task_c_backend_services",
    "phase1_task_e_https_tls",
    "phase1_task_g_protocol_flow_evidence",
    "phase1_task_f_http_caching",
    "phase2_extension_a_backup_dns",
    "phase2_extension_c_service_isolation",
    "phase2_extension_f_troubleshooting",
    "deliverable_phase2_final_report",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_one_to_one(
    expected_ids: set[str],
    predicted_matches: list[str | None],
    *,
    expected_categories: dict[str, str] | None = None,
    predicted_categories: list[str] | None = None,
) -> tuple[int, int, int]:
    """Score manually adjudicated semantic matches without rewarding merges."""
    seen: set[str] = set()
    true_positives = 0
    false_positives = 0

    for index, expected_id in enumerate(predicted_matches):
        if expected_id is None or expected_id not in expected_ids or expected_id in seen:
            false_positives += 1
            continue
        if expected_categories is not None and predicted_categories is not None:
            if predicted_categories[index] != expected_categories[expected_id]:
                false_positives += 1
                continue
        seen.add(expected_id)
        true_positives += 1

    false_negatives = len(expected_ids - seen)
    return true_positives, false_positives, false_negatives


def _is_supported(value: str | None, evidence: str | None, source: str) -> bool:
    if not value or not evidence:
        return False
    normalize = lambda text: " ".join(text.casefold().split())
    normalized_source = normalize(source)
    normalized_evidence = normalize(evidence)
    return normalized_evidence in normalized_source and normalize(value) in normalized_source


def test_ground_truth_fixture_is_separate_and_caveats_missing_pdf() -> None:
    ground_truth = _load_json(GROUND_TRUTH_PATH)
    archived_output = _load_json(ARCHIVED_OUTPUT_PATH)

    assert ground_truth["source"]["requested_pdf_found"] is False
    assert ground_truth["source"]["archived_analysis_title"] == "CN_Project_Doc.pdf"
    assert "equivalence" in ground_truth["source"]["scope_note"]
    assert archived_output["analysis_title"] != ground_truth["source"]["requested_filename"]
    assert len(ground_truth["expected_tasks"]) == 25
    assert len(archived_output["tasks"]) == 9
    assert len({task["id"] for task in ground_truth["expected_tasks"]}) == 25
    assert all(task["evidence"] for task in ground_truth["expected_tasks"])
    assert len(archived_output["summary"]) == 500
    assert archived_output["summary"].endswith("The evaluation is divided,")


def test_archived_task_metrics_use_one_to_one_semantic_matches() -> None:
    ground_truth = _load_json(GROUND_TRUTH_PATH)
    archived_output = _load_json(ARCHIVED_OUTPUT_PATH)
    expected_ids = {task["id"] for task in ground_truth["expected_tasks"]}

    true_positives, false_positives, false_negatives = _score_one_to_one(
        expected_ids,
        list(ARCHIVED_MATCHES),
    )

    assert len(archived_output["tasks"]) == len(ARCHIVED_MATCHES)
    assert (true_positives, false_positives, false_negatives) == (9, 0, 16)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / len(expected_ids)
    f1 = 2 * precision * recall / (precision + recall)
    assert precision == 1.0
    assert recall == pytest.approx(0.36)
    assert f1 == pytest.approx(0.5294117647)


def test_duplicate_predictions_count_as_false_positives() -> None:
    expected = {"task-a", "task-b", "task-c"}

    assert _score_one_to_one(expected, ["task-a", "task-a", "task-b"]) == (2, 1, 1)


def test_compound_prediction_cannot_match_multiple_expected_tasks() -> None:
    expected = {"private-lan", "private-dns"}
    one_compound_prediction = ["private-lan"]

    assert _score_one_to_one(expected, one_compound_prediction) == (1, 0, 1)


def test_wrong_classification_is_not_counted_as_a_match() -> None:
    expected = {"task-a"}

    assert _score_one_to_one(
        expected,
        ["task-a"],
        expected_categories={"task-a": "task"},
        predicted_categories=["blocker"],
    ) == (0, 1, 1)


@pytest.mark.parametrize(
    ("value", "evidence", "expected"),
    [
        ("2026-10-28", "Submit the report by 2026-10-28.", True),
        ("2027-10-28", "Submit the report by 2026-10-28.", False),
        ("Alice", "Submit the report by 2026-10-28.", False),
        ("2026-10-28", None, False),
    ],
)
def test_audit_rejects_unsupported_deadlines_owners_and_missing_evidence(
    value: str,
    evidence: str | None,
    expected: bool,
) -> None:
    source = "Submit the report by 2026-10-28."

    assert _is_supported(value, evidence, source) is expected


def test_empty_extraction_with_actionable_ground_truth_triggers_review() -> None:
    ground_truth = _load_json(GROUND_TRUTH_PATH)
    source = ground_truth["expected_tasks"][0]["evidence"][2]["quote"]
    empty_analysis = DocumentAnalysis(summary="No extracted items.")

    assert should_review_empty_extraction(source, empty_analysis)


def test_zero_events_and_blockers_are_valid_for_this_source() -> None:
    ground_truth = _load_json(GROUND_TRUTH_PATH)
    archived_output = _load_json(ARCHIVED_OUTPUT_PATH)

    assert ground_truth["expected_events"] == []
    assert ground_truth["expected_blockers"] == []
    assert archived_output["events"] == []
    assert archived_output["potential_blockers"] == []


def test_milestone_deadlines_are_not_invented_calendar_dates() -> None:
    ground_truth = _load_json(GROUND_TRUTH_PATH)
    dated_tasks = [
        task
        for task in ground_truth["expected_tasks"]
        if task["deadline_text"] is not None
    ]

    assert len(dated_tasks) == 7
    assert ground_truth["explicit_calendar_dates"] == []
    assert all(task["owner"] is None for task in ground_truth["expected_tasks"])
