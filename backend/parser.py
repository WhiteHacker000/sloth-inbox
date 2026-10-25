import json
from pathlib import Path


EMAIL_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "samples"
    / "emails.json"
)


def load_emails() -> list[dict]:
    with EMAIL_FILE.open("r", encoding="utf-8") as file:
        emails = json.load(file)

    if not isinstance(emails, list):
        raise ValueError("Expected a list of emails.")

    return emails