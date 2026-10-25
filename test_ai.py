from backend.ai import ask_qwen


if __name__ == "__main__":
    result = ask_qwen(
        "Summarize this email in one sentence: "
        "The frontend is complete, but backend testing remains."
    )
    print(result)