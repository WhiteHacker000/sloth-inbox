# Sloth AI 🦥

[![Tests](https://github.com/WhiteHacker000/sloth-inbox/actions/workflows/tests.yml/badge.svg)](https://github.com/WhiteHacker000/sloth-inbox/actions/workflows/tests.yml)

Sloth AI is a local-first document-intelligence assistant. It uses an open-weight model through Ollama to summarize plain-text documents and extract tasks, explicitly stated events, and potential blockers. Analyses, source text, and task status are saved in a local SQLite database.

**Do less. Get more done.**

## Implemented features

- Paste text or upload a UTF-8 `.txt`, selectable-text `.pdf`, or `.docx` file (up to the configured size limit).
- Analyze documents with a locally running Ollama model.
- Validate structured results with Pydantic and Ollama's JSON-schema response format.
- Review schema-valid empty results when the source contains action cues; retry one invalid model response with a corrective prompt, then report a recoverable error.
- View summaries, tasks, event dates as written, and potential blockers.
- Save, revisit, and inspect analyses and their original source documents.
- List tasks, filter by status, complete tasks, and reopen completed tasks.
- View document and task counts and recent analysis activity.
- Poll an explicitly configured folder for new or changed `.txt` documents in a separate worker process.
- Persist file hashes, processing status, retry timing, outcomes, and automation run history in SQLite.
- Review a database-backed daily briefing with task statuses, explicit full-date deadlines, recent analyses, and blockers.
- Keep inference local; no hosted-model fallback, telemetry, or document-content logging.

The existing email parser and email-analysis helper remain available as backend code, but email ingestion is not part of this document-analysis interface.

## Architecture

```text
app.py                         Streamlit dashboard
main.py                        Application entry point
backend/
  config.py                    Validated environment configuration
  document_service.py          In-memory TXT, PDF, and DOCX text extraction
  ai.py                        Ollama HTTP client and legacy email API
  prompts.py                   Untrusted-document analysis prompts
  schemas.py                   Pydantic analysis and task models
  analyzer.py                  Model orchestration and one bounded retry
  database.py                  SQLite repository and persistence
  task_service.py              Task status operations
  document_monitor.py          Stable-file polling and worker lifecycle
  worker.py                    Separate worker process entry point
  briefing.py                  Database-backed task/deadline summary
tests/                         Mocked unit and workflow tests
data/
  samples/emails.json          Existing sample email data
  sloth_ai.db                  Created locally on first launch
```

The document service, analyzer, model client, and repository are independent of Streamlit. PDF and DOCX extraction is available for manual uploads; automatic folder processing remains `.txt` only. OCR, email ingestion, and file modification are not implemented.

## Requirements

- Python 3.12 or newer (the current project environment was verified with Python 3.14).
- Ollama installed and running locally.
- A compatible Ollama model installed manually. The default is `qwen2.5:3b`.
- Hardware sufficient to run your selected model. Memory and speed requirements vary by model and quantization.

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Install Ollama from [ollama.com](https://ollama.com/), then start its local service. In a separate terminal, download the configured model:

```bash
ollama serve
ollama pull qwen2.5:3b
```

Model downloads are explicit and are never initiated by the application. If you already have another compatible model, set `OLLAMA_MODEL` in `.env` to that installed model instead.

## Configuration

Edit `.env` in the project root:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server base URL |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Installed model name (for example, `qwen3:4b`) |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Maximum duration for one model request |
| `MAX_DOCUMENT_SIZE_MB` | `5` | Maximum UTF-8 document size |
| `DATABASE_PATH` | `./data/sloth_ai.db` | Local SQLite database file |
| `DOCUMENT_INPUT_DIR` | empty | Optional existing folder to poll; can also be configured in the Automation view |
| `DOCUMENT_POLL_INTERVAL_SECONDS` | `30` | Delay between worker scans |
| `DOCUMENT_STABILITY_SECONDS` | `2` | Time file size and modification time must remain unchanged before reading |
| `DOCUMENT_RECURSIVE` | `false` | Scan subfolders only when explicitly set to `true` |
| `DOCUMENT_MAX_ATTEMPTS` | `3` | Maximum total attempts per unchanged content hash |

Settings are validated at startup. The application checks Ollama and the configured model and displays an actionable message if either is unavailable. It does not automatically download a missing model. A model/API error or invalid model output is shown as a failure; no fabricated result is saved.

## Run

```bash
streamlit run app.py
```

Alternatively:

```bash
python main.py
```

The app binds to `127.0.0.1` by default and is available only on the local machine. Use the local URL printed by Streamlit. Use **Analyze document** to upload a `.txt`, `.pdf`, or `.docx` file or paste text. Uploaded files are size-checked and extracted in memory; a text preview is shown before analysis. PDF extraction requires selectable text—scanned PDFs that need OCR are not supported. DOCX extraction includes paragraphs and tables in document order. The original upload filename is used as the default analysis title. After analysis, inspect the Overview, Daily briefing, Tasks, and History sections. Task changes update the task record while preserving the original extracted analysis.

## Deployment scope

The supported setup runs Streamlit, Ollama, and SQLite locally on the same
machine. The app uses local Ollama inference by default and does not fall back
to a hosted AI service. Public or cloud deployment is not supported by this
project: authentication, remote inference, and remote persistent storage are
not provided. Do not expose the Ollama API or the unauthenticated Streamlit app
to the public internet.

## Automatic folder processing

In the **Automation** view, enter and save an existing folder path, or set `DOCUMENT_INPUT_DIR` in `.env`. A folder saved in the UI is stored in local SQLite and takes precedence over `.env`; use **Use folder from .env instead** to remove the saved override. The configured directory is never created automatically. Only `.txt` files are supported. Subdirectories are ignored unless `DOCUMENT_RECURSIVE=true`.

Start the worker in a separate terminal from the project root:

```bash
source .venv/bin/activate
python -m backend.worker
```

Stop it with **Ctrl+C** in that terminal. The worker is separate from Streamlit, so UI reruns do not spawn threads and closing the worker process stops polling. It does not continue after the process or computer shuts down unless you separately configure an OS service (not implemented here).

The worker waits until a file's size and modification time have been stable for the configured settle period, reads it through the same UTF-8 and size validator as manual uploads, and checks its SHA-256 content hash. Unchanged content is not analyzed twice, even if its modification time changes. Changed content creates a new analysis; earlier analyses and tasks are retained. Paths are stored relative to the watched folder, so same-named files in different subdirectories have separate records. Symlinks are skipped. Source files are never moved or modified.

Ollama connection/HTTP and invalid structured-output failures are retried with exponential backoff, up to `DOCUMENT_MAX_ATTEMPTS`. Invalid encodings, oversized files, and unreadable files are recorded and are not retried until the content or file metadata changes. Retry timing and outcomes are stored in SQLite. SQLite claims prevent two workers from processing the same file concurrently. The Automation view shows the worker heartbeat, file states, recent runs, and failed files.

## Daily briefing

The **Daily briefing** calculates task counts from SQLite. It shows pending and completed tasks, recent analyses and recorded blockers, activity totals, and deadlines with explicit full dates. ISO dates and written month/day/year dates can be classified as upcoming or overdue. Relative wording (such as “tomorrow”), weekday-only values, yearless dates, and ambiguous numeric dates remain unresolved; original deadline text is preserved. The app does not ask the model to compute statistics or guess dates. Task source-analysis identifiers can be opened from Tasks or History. When an analysis has no extracted tasks, events, or blockers, the app shows a review notice; empty results may be valid, but the model may also have missed relevant information.

## Database and privacy

By default, Sloth AI creates `data/sloth_ai.db` when it starts. It stores analysis results, extracted tasks, task status, and original source text so a saved analysis can be revisited. Automatic processing additionally stores the configured folder path, relative filenames, file hashes, sizes, timestamps, processing outcomes, and error categories. Choose a secure local location with `DATABASE_PATH` if the source documents are sensitive. The database and SQLite sidecar files are excluded from Git.

Document text is sent only to the configured Ollama endpoint. With the default URL, that endpoint is on the same machine. The application does not call hosted AI APIs, send analytics, log document bodies/prompts, execute document instructions, invoke model tools, or modify user files. Manual analysis reads only a document explicitly selected or pasted by the user; PDF/DOCX parsing occurs locally in memory. The separate worker reads only `.txt` files in the configured folder. Local execution is not itself a security guarantee; protect the device, Ollama endpoint, input folder, and database appropriately.

Initial model downloads require an internet connection. After setup, inference can run locally, but this project has not been certified for fully offline operation.

## Tests

Tests mock model calls and do not require an Ollama process:

```bash
python -m pytest -q
```

Coverage includes TXT/PDF/DOCX extraction and upload workflows, empty-result feedback, configuration validation, model availability and HTTP failures, malformed structured output and its bounded retry, input validation, persistence and rollback, task status changes, analysis history, polling and hash de-duplication, changed files, write stability, bounded retries, duplicate claims, worker shutdown, email-analysis compatibility, and deadline classification. The automated tests mock AI interactions and do not require Ollama.

The GitHub Actions workflow runs the tests and compilation check with Python
3.14 for pushes to `main` and pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the contribution workflow and project-specific privacy requirements. This
repository does not currently include a software license; contact the
maintainers before redistributing or reusing the code.

## Troubleshooting

- **Ollama is unavailable:** Start Ollama (for example, `ollama serve`) and verify `OLLAMA_BASE_URL`.
- **Model not found:** Run `ollama pull <model-name>` or update `OLLAMA_MODEL` to an installed model. Startup never downloads it automatically.
- **Request timed out:** Increase `OLLAMA_TIMEOUT_SECONDS` or use a smaller model.
- **Invalid analysis output:** The app makes one schema-guided correction attempt. If that also fails, retry or choose another model; no invalid analysis is persisted.
- **Document upload rejected:** Upload a UTF-8 `.txt`, `.pdf`, or `.docx` file within the configured size limit. Scanned PDFs need OCR, which is not supported.
- **Database cannot be opened:** Check that `DATABASE_PATH` points to a writable directory and that the file is not locked or corrupted.
- **Python imports fail:** Run commands from the project root with the project virtual environment activated.

## Open-weight model rationale and trade-offs

Local inference can keep document content on the user's device, allows a compatible model to be replaced independently from most application logic, and avoids per-request hosted AI API charges. The user controls the selected model and inference setup. Local inference may continue without internet after the runtime and model are installed, subject to the environment.

Local models consume system memory and compute; smaller models may extract information less accurately; model updates and downloads are the user's responsibility; and each model has its own license and usage terms. Structured responses are still validated, but schema validity does not prove factual accuracy. Review important extracted tasks, dates, and blockers.

## Known limitations and roadmap

Automatic processing is a lightweight poller, not an always-on OS service. It runs only while its separate worker process is alive. Abrupt termination can leave a worker status marked running until its heartbeat is stale. The settle check uses filesystem size and modification time; filesystems that do not update those reliably can delay change detection. Automatic processing supports `.txt` only. The briefing leaves relative and ambiguous dates unresolved. OCR, cross-document search, email ingestion, external integrations, and file organization are not implemented. Sloth AI does not automatically create calendar events or perform actions.

Possible future work includes Markdown ingestion, document classification and cross-document summaries, inbox processing, preview-based file organization, scheduled workflows, additional local model providers, and retrieval-augmented generation. Any consequential automation should require explicit user approval and offer a recoverable preview.

## Model licenses

Ollama distributes model packages, but the license and permitted uses depend on the selected model and its upstream publisher. Check the model card and license for the exact model and version you download; the Sloth AI project does not grant rights to model weights.