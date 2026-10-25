from backend.parser import load_emails
from backend.analyzer import analyze_email


emails = load_emails()

for email in emails:
    print("\n" + "=" * 50)
    print("Subject:", email["subject"])

    result = analyze_email(email)

    print("Summary:", result["summary"])
    print("Category:", result["category"])
    print("Priority:", result["priority"])
    print("Requires reply:", result["requires_reply"])
    print("Deadlines:", result["deadlines"])
    print("Action items:", result["action_items"])