from backend.ai import ask_qwen

result = ask_qwen(
    "Summarize this email in one sentence: "
    "The frontend is complete, but backend testing remains."
)

print(result)