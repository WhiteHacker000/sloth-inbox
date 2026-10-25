# Contributing to Sloth AI

Thanks for helping improve Sloth AI. Contributions should preserve its local-first
design: document content stays with the user, and model inference uses the
configured Ollama service unless the maintainers explicitly approve an architectural
change.

## Before opening a pull request

1. Check existing issues and pull requests to avoid duplicate work.
2. For a larger behavior change, open an issue first to agree on scope.
3. Keep changes focused and include deterministic tests for behavior changes.
4. Do not commit credentials, `.env` files, databases, virtual environments,
   private documents, model files, or real user data.
5. Do not add hosted inference, expose Ollama publicly, or add telemetry without
   explicit maintainer approval.

## Development and tests

Use Python 3.14 and install dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the same checks used by CI:

```bash
python -m pytest -q
python -m compileall -q app.py main.py backend tests test_ai.py
```

The test suite mocks Ollama requests and does not need a running Ollama server.
For an interactive app run, install Ollama separately and follow the setup steps
in the README.

## Pull requests

- Describe the user-facing behavior and any relevant design trade-offs.
- Include test results and note any manual checks.
- Keep document fixtures synthetic or public and ensure they contain no private
  information.
- Update the README when setup, configuration, limitations, or deployment
  behavior changes.

The repository currently has no declared software license. Please contact the
maintainers before redistributing or incorporating this code into another
project.
