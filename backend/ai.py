import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"


def ask_qwen(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "think": False,
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180,
        )

        response.raise_for_status()
        result = response.json()

        # Debugging information
        print("\n--- OLLAMA DEBUG ---")
        print("Model:", result.get("model"))
        print("Done:", result.get("done"))
        print("Response:", repr(result.get("message", {}).get("content")))
        print("--------------------\n")

        content = result.get("message", {}).get("content", "")

        if not content.strip():
            raise ValueError(
                "Ollama returned an empty response. "
                "Check the model's thinking/output behavior."
            )

        return content.strip()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not communicate with Ollama: {exc}"
        ) from exc