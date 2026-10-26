import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from backend.ai import (
    OllamaClient,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from backend.analyzer import (
    AnalysisOutputError,
    analyze_document,
)
from backend.briefing import build_daily_briefing
from backend.config import ConfigurationError, Settings
from backend.database import DatabaseError, DocumentRepository
from backend.document_monitor import (
    DocumentMonitor,
    MonitorConfigurationError,
    directory_identifier,
    resolve_input_directory,
)
from backend.document_service import (
    DocumentError,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    extract_document_text,
    validate_text,
)
from backend.schemas import DocumentAnalysis, TaskStatus
from backend.task_service import TaskService

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
MODEL_OPTIONS = {
    "qwen2.5:3b": "Qwen 2.5 3B",
    "gemma3:1b": "Gemma 3 1B",
}

st.set_page_config(
    page_title="Sloth AI",
    page_icon="🦥",
    layout="wide",
)
style_path = Path(__file__).parent / ".streamlit" / "dashboard.css"
st.markdown(
    f"<style>{style_path.read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)
st.title("Sloth AI", icon="🦥")
st.caption("Do less. Get more done. Your documents stay on this device.")

try:
    settings = Settings.from_env()
    repository = DocumentRepository(settings.database_path)
    task_service = TaskService(repository)
except ConfigurationError as exc:
    st.error(f"Configuration error: {exc}")
    st.info("Review your .env file and compare it with .env.example.")
    st.stop()
except DatabaseError as exc:
    logger.error("Database initialization failed: %s", type(exc).__name__)
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Local Model")
    available_model_tags = list(MODEL_OPTIONS)
    if settings.ollama_model not in MODEL_OPTIONS:
        available_model_tags.append(settings.ollama_model)
    selected_model = st.selectbox(
        "Ollama model",
        options=available_model_tags,
        index=available_model_tags.index(settings.ollama_model),
        format_func=lambda model: MODEL_OPTIONS.get(
            model,
            f"Other (configured): {model}",
        ),
        key="ollama-model",
        help="Choose the local model used for document analysis.",
    )
    client = OllamaClient(replace(settings, ollama_model=selected_model))
    st.caption(
        "The automatic folder worker uses OLLAMA_MODEL from the environment."
    )
    try:
        client.ensure_model_available()
    except OllamaModelNotFoundError:
        st.warning(
            "Ollama is running, but the configured model is unavailable.",
            icon=":material/download:",
        )
    except OllamaUnavailableError:
        st.warning(
            "Ollama is unavailable. Start the local service and try again.",
            icon=":material/cloud_off:",
        )
    except OllamaError:
        st.warning(
            "The local model could not be checked. Review the Ollama service setup.",
            icon=":material/warning:",
        )
    else:
        st.success(
            "Ollama is available and the configured model is installed.",
            icon=":material/check_circle:",
        )

overview_tab, briefing_tab, analyze_tab, tasks_tab, history_tab, automation_tab = st.tabs(
    [
        "Overview",
        "Daily briefing",
        "Analyze document",
        "Tasks",
        "History",
        "Automation",
    ]
)


def show_analysis(analysis: DocumentAnalysis) -> None:
    st.subheader("Summary")
    st.write(analysis.summary)

    left, right = st.columns(2)
    with left:
        st.subheader(f"Tasks ({len(analysis.tasks)})")
        if analysis.tasks:
            for task in analysis.tasks:
                with st.container(border=True):
                    st.markdown(f"**{task.description}**")
                    st.caption(f"Status · {task.status.replace('_', ' ').title()}")
                    if task.owner:
                        st.caption(f"Owner · {task.owner}")
                    if task.deadline_text:
                        st.caption(f"Deadline · {task.deadline_text}")
                    if task.priority:
                        st.caption(f"Priority · {task.priority.title()}")
                    if task.evidence:
                        st.caption(f"Source evidence · “{task.evidence}”")
        else:
            st.info("No actionable tasks were identified.")

    with right:
        st.subheader(f"Events ({len(analysis.events)})")
        if analysis.events:
            for event in analysis.events:
                with st.container(border=True):
                    st.markdown(f"**{event.description}**")
                    st.caption(f"Date as written · {event.date_text}")
                    if event.evidence:
                        st.caption(f"Source evidence · “{event.evidence}”")
        else:
            st.info("No explicitly dated events were identified.")

    st.subheader(f"Potential blockers ({len(analysis.potential_blockers)})")
    if analysis.potential_blockers:
        for blocker in analysis.potential_blockers:
            st.warning(blocker, icon=":material/warning:")
    else:
        st.info("No potential blockers were identified.")


with overview_tab:
    try:
        analyses = repository.list_analyses()
        analysis_count = repository.analysis_count()
        counts = repository.task_counts()
    except DatabaseError:
        logger.error("Could not load overview data")
        st.error("Overview data could not be loaded from the local database.")
    else:
        metrics = st.columns(4)
        metrics[0].metric("Documents analyzed", analysis_count)
        metrics[1].metric("Tasks extracted", sum(counts.values()))
        metrics[2].metric("Pending tasks", counts["pending"])
        metrics[3].metric("Completed tasks", counts["completed"])
        st.subheader("Recent activity")
        if analyses:
            st.dataframe(
                [
                    {
                        "Title": item["title"] or "Untitled document",
                        "Summary": item["summary"],
                        "Tasks": item["task_count"],
                        "Analyzed": item["created_at"],
                    }
                    for item in analyses[:5]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No documents analyzed yet. Add a text document to get started.")

with briefing_tab:
    st.subheader("Daily productivity briefing")
    try:
        briefing = build_daily_briefing(repository)
    except DatabaseError:
        logger.error("Could not build productivity briefing")
        st.error("The briefing could not be loaded from the local database.")
    else:
        activity = briefing["activity"]
        counts = briefing["counts"]
        with st.container(horizontal=True):
            st.metric("Pending tasks", counts["pending"] + counts["in_progress"])
            st.metric("Completed tasks", counts["completed"])
            st.metric("Overdue", len(briefing["overdue_tasks"]))
            st.metric("Analyzed today", activity["analyses_today"])
        st.caption(
            f"Briefing for {briefing['today']}. Counts and statuses come from SQLite."
        )

        deadline_columns = st.columns(3)
        with deadline_columns[0]:
            st.markdown("**Upcoming deadlines**")
            if briefing["upcoming_deadlines"]:
                for task in briefing["upcoming_deadlines"]:
                    st.write(
                        f"- {task['description']} — {task['deadline_text']} "
                        f"({task['source_title'] or 'Untitled document'})"
                    )
            else:
                st.info("No upcoming explicit deadlines.")
        with deadline_columns[1]:
            st.markdown("**Overdue tasks**")
            if briefing["overdue_tasks"]:
                for task in briefing["overdue_tasks"]:
                    st.write(
                        f"- {task['description']} — {task['deadline_text']} "
                        f"({task['source_title'] or 'Untitled document'})"
                    )
            else:
                st.info("No overdue tasks with an unambiguous full date.")
        with deadline_columns[2]:
            st.markdown("**Unresolved deadlines**")
            if briefing["unresolved_deadlines"]:
                for task in briefing["unresolved_deadlines"]:
                    st.write(
                        f"- {task['description']} — {task['deadline_text']} "
                        f"({task['source_title'] or 'Untitled document'})"
                    )
            else:
                st.info("No unresolved deadline text.")

        with st.expander("Pending tasks"):
            if briefing["pending_tasks"]:
                for task in briefing["pending_tasks"]:
                    st.markdown(
                        f"**{task['description']}** — "
                        f"{task['deadline_text'] or 'No deadline stated'}"
                    )
                    st.caption(
                        f"Source: {task['source_title'] or 'Untitled document'} "
                        f"(analysis {task['source_document_id']})"
                    )
            else:
                st.info("There are no pending tasks.")
        with st.expander("Recently completed tasks"):
            if briefing["completed_tasks"]:
                for task in briefing["completed_tasks"]:
                    st.markdown(f"**{task['description']}**")
                    st.caption(
                        f"Source: {task['source_title'] or 'Untitled document'} "
                        f"(analysis {task['source_document_id']})"
                    )
            else:
                st.info("No tasks have been completed yet.")

        st.subheader("Recent document activity")
        if briefing["recent_analyses"]:
            for analysis in briefing["recent_analyses"]:
                with st.container(border=True):
                    st.markdown(f"**{analysis['title'] or 'Untitled document'}**")
                    st.caption(f"Analyzed: {analysis['created_at']}")
                    st.write(analysis["summary"])
                    st.caption(f"Analysis ID: {analysis['id']}")
        else:
            st.info("No document analyses were saved in the recent period.")

        st.subheader("Potential blockers from recent analyses")
        if briefing["potential_blockers"]:
            for blocker in briefing["potential_blockers"]:
                st.write(
                    f"- {blocker['description']} "
                    f"({blocker['title'] or 'Untitled document'}; "
                    f"analysis {blocker['analysis_id']})"
                )
        else:
            st.info("No blockers were recorded in recent analyses.")

        st.subheader("Activity statistics")
        st.write(
            f"All-time analyses: {activity['analyses']} · "
            f"all-time tasks: {activity['tasks']} · "
            f"successful automatic runs: {activity['successful_runs']} · "
            f"failed automatic runs: {activity['failed_runs']} · "
            f"automatic runs today: {activity['runs_today']}"
        )

with analyze_tab:
    st.subheader("Analyze a document")
    st.caption(
        "Upload a .txt, .pdf, or .docx file, or paste text. Analysis uses only "
        "the configured local Ollama model. Scanned PDFs that need OCR are not "
        "supported yet."
    )
    document_title = st.text_input("Document title (optional)")
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .pdf, or .docx)",
        type=[extension.lstrip(".") for extension in SUPPORTED_DOCUMENT_EXTENSIONS],
        help=f"Maximum size: {settings.max_document_size_mb} MB.",
        max_upload_size=settings.max_document_size_mb,
    )
    uploaded_text: str | None = None
    upload_error: str | None = None
    if uploaded_file is not None:
        try:
            if uploaded_file.size > settings.max_document_size_bytes:
                raise DocumentError("The document exceeds the configured size limit.")
            uploaded_text = extract_document_text(
                uploaded_file.getvalue(),
                uploaded_file.name,
                settings.max_document_size_bytes,
            )
        except DocumentError as exc:
            upload_error = str(exc)
            st.error(upload_error)
        else:
            with st.expander("Preview extracted text"):
                st.text(uploaded_text[:4000])
                if len(uploaded_text) > 4000:
                    st.caption("Preview truncated; the full extracted text will be analyzed.")

    pasted_text = st.text_area(
        "Or paste document text",
        height=220,
        placeholder="Paste the text you want Sloth AI to summarize and analyze...",
    )

    if st.button("Analyze document", type="primary"):
        try:
            if uploaded_file is not None:
                if pasted_text.strip():
                    raise DocumentError(
                        "Choose either an uploaded file or pasted text, not both."
                    )
                if upload_error is not None or uploaded_text is None:
                    raise DocumentError(upload_error or "The uploaded document could not be extracted.")
                source_text = uploaded_text
                title = document_title.strip() or uploaded_file.name
            else:
                source_text = validate_text(
                    pasted_text,
                    settings.max_document_size_bytes,
                )
                title = document_title.strip() or None

            client.ensure_model_available()
            logger.info("Document analysis started")
            with st.spinner("Analyzing locally with Ollama..."):
                result = analyze_document(source_text, client, title)
            analysis_id = repository.save_analysis(source_text, result, title)
            st.session_state["latest_analysis_id"] = analysis_id
            logger.info("Document analysis completed id=%s", analysis_id)
            if not (
                result.tasks or result.events or result.potential_blockers
            ):
                st.warning(
                    "No tasks, events, or potential blockers were extracted. The "
                    "document may contain no actionable content, or the model may "
                    "have missed relevant information. Review the source and summary.",
                    title="Review extraction",
                )
            else:
                st.success("Analysis completed and saved locally.")
        except (DocumentError, OllamaError, AnalysisOutputError, DatabaseError) as exc:
            logger.warning("Document analysis failed category=%s", type(exc).__name__)
            st.error(str(exc))

    latest_analysis_id = st.session_state.get("latest_analysis_id")
    if latest_analysis_id:
        try:
            saved = repository.get_analysis(latest_analysis_id)
        except (DatabaseError, LookupError) as exc:
            logger.error("Could not reopen recent analysis category=%s", type(exc).__name__)
            st.error("The saved analysis could not be reopened.")
        else:
            st.divider()
            st.subheader(saved["title"] or "Analysis result")
            show_analysis(saved["analysis"])

with tasks_tab:
    st.subheader("Extracted tasks")
    status_filters: dict[str, TaskStatus | None] = {
        "All statuses": None,
        "Pending": "pending",
        "In progress": "in_progress",
        "Completed": "completed",
    }
    selected_status = st.selectbox(
        "Filter by status",
        list(status_filters),
    )
    try:
        tasks = task_service.list_tasks(status_filters[selected_status])
    except DatabaseError:
        logger.error("Could not load tasks")
        st.error("Tasks could not be loaded from the local database.")
    else:
        if not tasks:
            st.info("No tasks match this filter.")
        for task in tasks:
            with st.expander(
                f"{task['description']} — {task['status'].replace('_', ' ').title()}"
            ):
                st.write(f"**Source:** {task['source_title'] or 'Untitled document'}")
                st.write(f"**Owner:** {task['owner'] or 'Not specified'}")
                st.write(f"**Deadline:** {task['deadline_text'] or 'Not specified'}")
                st.write(f"**Priority:** {task['priority'] or 'Not specified'}")
                if task["evidence"]:
                    st.caption(f"Evidence: “{task['evidence']}”")
                try:
                    source = task_service.source_analysis(task["source_document_id"])
                except (DatabaseError, LookupError):
                    st.warning("The source analysis is unavailable.")
                else:
                    st.caption(f"Source summary: {source['analysis'].summary}")
                if task["status"] == "completed":
                    if st.button("Reopen task", key=f"reopen-{task['id']}"):
                        try:
                            task_service.reopen(task["id"])
                        except (DatabaseError, LookupError):
                            logger.error("Could not reopen task")
                            st.error("This task could not be reopened.")
                        else:
                            st.rerun()
                else:
                    if st.button("Mark completed", key=f"complete-{task['id']}"):
                        try:
                            task_service.complete(task["id"])
                        except (DatabaseError, LookupError):
                            logger.error("Could not complete task")
                            st.error("This task could not be marked complete.")
                        else:
                            st.rerun()

with history_tab:
    st.subheader("Analysis history")
    try:
        analyses = repository.list_analyses()
    except DatabaseError:
        logger.error("Could not load analysis history")
        st.error("Saved analyses could not be loaded from the local database.")
    else:
        if not analyses:
            st.info("Saved analyses will appear here.")
        else:
            selected_analysis_id = st.selectbox(
                "Select an analysis",
                [item["id"] for item in analyses],
                format_func=lambda analysis_id: next(
                    (
                        f"{item['title'] or 'Untitled document'} — {item['created_at']}"
                        for item in analyses
                        if item["id"] == analysis_id
                    ),
                    analysis_id,
                ),
            )
            try:
                saved = repository.get_analysis(selected_analysis_id)
            except (DatabaseError, LookupError):
                logger.error("Could not load selected history item")
                st.error("This saved analysis could not be loaded.")
            else:
                st.caption(f"Analyzed: {saved['created_at']}")
                show_analysis(saved["analysis"])
                with st.expander("View original document"):
                    st.text(saved["source_text"])

with automation_tab:
    st.subheader("Automatic document processing")
    st.caption(
        "Only the explicitly selected folder is watched. Plain-text .txt files "
        "are read; source files are never moved or changed."
    )
    saved_directory = repository.get_setting("document_input_dir")
    configured_directory = (
        saved_directory
        or (str(settings.document_input_dir) if settings.document_input_dir else "")
    )
    with st.form("watched-folder-form"):
        folder_value = st.text_input(
            "Document input folder",
            value=configured_directory,
            help="Enter an existing folder path. Subdirectories are ignored by default.",
        )
        save_folder = st.form_submit_button("Save watched folder")
    if save_folder:
        candidate = Path(folder_value).expanduser()
        if not folder_value.strip():
            st.error("Choose a folder path before saving.")
        elif not candidate.exists() or not candidate.is_dir():
            st.error("The selected folder must already exist and be a directory.")
        else:
            try:
                repository.set_setting("document_input_dir", str(candidate.resolve()))
            except DatabaseError:
                logger.error("Could not save watched folder configuration")
                st.error("The watched folder could not be saved.")
            else:
                st.success("Watched folder saved locally.")
                st.rerun()
    if saved_directory and st.button("Use folder from .env instead"):
        try:
            repository.delete_setting("document_input_dir")
        except DatabaseError:
            logger.error("Could not reset watched folder configuration")
            st.error("The watched folder configuration could not be reset.")
        else:
            st.rerun()

    effective_directory = None
    try:
        effective_directory = resolve_input_directory(settings, repository)
        monitor = DocumentMonitor(
            effective_directory,
            repository,
            client,
            settings,
        )
    except (MonitorConfigurationError, DatabaseError, OSError) as exc:
        st.info(
            "Automatic processing is not configured yet. Choose an existing folder "
            "above or set DOCUMENT_INPUT_DIR in .env."
        )
        logger.info("Watched folder unavailable category=%s", type(exc).__name__)
    else:
        st.write(f"**Watched folder:** `{effective_directory}`")
        st.write(
            f"**File types:** `.txt` · **Subdirectories:** "
            f"{'included' if settings.document_recursive else 'not scanned'} · "
            f"**Polling:** {settings.document_poll_interval_seconds:g} seconds · "
            f"**Write-settle period:** {settings.document_stability_seconds:g} seconds"
        )
        st.code("python -m backend.worker", language="bash")
        st.caption(
            "Start this command in a separate terminal to run the worker. "
            "Press Ctrl+C in that terminal to stop it. The worker stops when its "
            "process exits; it does not run while the application is closed."
        )

        try:
            worker_status = repository.get_worker_status(monitor.directory_id)
            recent_runs = repository.list_processing_runs(
                monitor.directory_id, limit=20
            )
            failed_files = repository.list_processing_files(
                monitor.directory_id, limit=100, status="failed"
            )
            file_counts = repository.processing_file_counts(monitor.directory_id)
            processing_file_count = repository.count_processing_runs(
                monitor.directory_id
            )
            file_statuses = repository.list_processing_files(
                monitor.directory_id, limit=1
            )
        except DatabaseError:
            logger.error("Could not load automatic processing status")
            st.error("Automatic processing history could not be loaded.")
        else:
            status_is_live = False
            if worker_status and worker_status["status"] == "running":
                heartbeat = datetime.fromisoformat(str(worker_status["heartbeat_at"]))
                status_is_live = (
                    datetime.now(UTC) - heartbeat
                ).total_seconds() <= settings.document_poll_interval_seconds * 3 + 15
            if status_is_live:
                st.success(
                    f"Worker running (PID {worker_status['pid']}); last heartbeat "
                    f"{worker_status['heartbeat_at']}."
                )
            elif worker_status:
                st.warning("Worker is stopped or its last heartbeat is stale.")
            else:
                st.info("Worker has not been started for this folder.")

            st.write(
                f"Tracked files: {sum(file_counts.values())} · "
                f"successfully processed: {file_counts['succeeded']} · "
                f"processing attempts: {processing_file_count} · "
                f"failed files: {file_counts['failed']}"
            )
            if not file_statuses:
                st.info("No .txt files have been discovered in this folder yet.")
            if failed_files:
                st.subheader("Failed processing jobs")
                st.dataframe(
                    [
                        {
                            "File": item["relative_path"],
                            "Error category": item["error_category"],
                            "Attempts": item["retry_count"],
                            "Last updated": item["updated_at"],
                        }
                        for item in failed_files
                    ],
                    hide_index=True,
                    width="stretch",
                )
            st.subheader("Recent automation runs")
            if recent_runs:
                st.dataframe(
                    [
                        {
                            "File": item["relative_path"],
                            "Outcome": item["outcome"],
                            "Error category": item["error_category"],
                            "Started": item["started_at"],
                            "Finished": item["finished_at"],
                        }
                        for item in recent_runs
                    ],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("Automatic processing runs will appear here.")