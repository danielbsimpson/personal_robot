"""
Personal Robot — main conversation loop.

Usage:
    .venv\\Scripts\\python.exe src/main.py

Commands during conversation:
    quit / exit / q    — exit cleanly
    Ctrl+C             — interrupt and exit cleanly
"""

import sys

from src.llm.client import OllamaClient, trim_history
from src.llm.prompts import BASE_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "phi4-mini"
# Drop oldest history pairs when total content exceeds this many characters
# (~4,096 tokens × ~4 chars/token)
CONTEXT_LIMIT_CHARS = 4096 * 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_banner() -> None:
    print("=" * 60)
    print("  Personal Robot — Local LLM Conversation")
    print(f"  Model: {MODEL}")
    print("  Type 'quit' or press Ctrl+C to exit.")
    print("=" * 60)
    print()


def get_user_input(prompt: str = "You: ") -> str:
    """Read a line from stdin, stripping whitespace."""
    try:
        return input(prompt).strip()
    except EOFError:
        return "quit"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print_banner()

    client = OllamaClient(
        model=MODEL,
        system_prompt=BASE_SYSTEM_PROMPT,
    )

    if not client.is_available():
        print("[ERROR] Cannot reach Ollama at http://localhost:11434")
        print("        Make sure Ollama is running: start the Ollama app or run 'ollama serve'")
        sys.exit(1)

    print("[OK] Ollama is running. Starting conversation...\n")

    conversation: list[dict] = []

    while True:
        user_text = get_user_input("You: ")

        if not user_text:
            continue

        if user_text.lower() in {"quit", "exit", "q"}:
            print("\nGoodbye!")
            break

        # Add user message to history
        conversation.append({"role": "user", "content": user_text})

        # Trim history to stay within context budget
        conversation = trim_history(conversation, limit_chars=CONTEXT_LIMIT_CHARS)

        # Get response (streaming — tokens print as they arrive)
        print("Robot: ", end="", flush=True)
        try:
            response = client.chat(conversation, stream=True)
        except Exception as e:
            print(f"\n[ERROR] LLM call failed: {e}")
            # Remove the user message we just added so history stays clean
            conversation.pop()
            continue

        # Add assistant response to history
        conversation.append({"role": "assistant", "content": response})
        print()  # blank line between turns


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
