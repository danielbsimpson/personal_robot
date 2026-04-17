"""
Personal Robot — main conversation loop.

Usage:
    .venv\\Scripts\\python.exe src/main.py

Commands during conversation:
    quit / exit / q    — exit cleanly
    Ctrl+C             — interrupt and exit cleanly
"""

import sys

from src.llm.client import OllamaClient, trim_history, OLLAMA_BASE_URL
from src.llm.context import ContextBudget
from src.llm.prompts import BASE_SYSTEM_PROMPT, get_time_section
from src.memory.soul import SoulFile, maybe_update_soul, maybe_grow_curiosity, SOUL_UPDATE_EVERY, SOUL_CURIOSITY_EVERY
from src.memory.vector_store import MemoryStore
from src.memory.summariser import summarise_session
from src.utils.log import ConversationLogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "phi4-mini"
BUDGET = ContextBudget()  # defaults: 4096 tokens, 512 reserve


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


def _save_session_summary(
    conversation: list[dict],
    store: MemoryStore,
) -> None:
    """Summarise the session and persist it to long-term memory if non-empty."""
    summary = summarise_session(conversation, MODEL, OLLAMA_BASE_URL)
    if summary:
        store.add_memory(summary, {"source": "session_summary"})
        print("[Memory] Session summary saved.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print_banner()

    # Load soul file and inject into system prompt
    soul = SoulFile()
    soul_section = soul.to_prompt_section(budget_chars=BUDGET.soul_budget_chars())
    system_prompt = BASE_SYSTEM_PROMPT
    if soul_section:
        system_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{soul_section}"
    time_section = get_time_section()
    if time_section:
        system_prompt = f"{system_prompt}\n\n{time_section}"

    client = OllamaClient(
        model=MODEL,
        system_prompt=system_prompt,
    )

    if not client.is_available():
        print("[ERROR] Cannot reach Ollama at http://localhost:11434")
        print("        Make sure Ollama is running: start the Ollama app or run 'ollama serve'")
        sys.exit(1)

    print("[OK] Ollama is running. Starting conversation...\n")

    conversation: list[dict] = []
    message_count = 0
    conv_logger = ConversationLogger(model=MODEL)
    memory_store = MemoryStore()

    try:
        while True:
            user_text = get_user_input("You: ")

            if not user_text:
                continue

            if user_text.lower() in {"quit", "exit", "q"}:
                print("\nGoodbye!")
                break

            # Add user message to history
            conversation.append({"role": "user", "content": user_text})
            conv_logger.log_turn("user", user_text)

            # RAG injection — query long-term memory for context relevant to this message
            rag_results = memory_store.query_memory(user_text)
            current_system_prompt = system_prompt
            if rag_results:
                rag_budget = BUDGET.rag_budget_chars()
                kept, total_chars = [], 0
                for result in rag_results:
                    entry = f"- {result}"
                    if total_chars + len(entry) + 1 > rag_budget:
                        break
                    kept.append(entry)
                    total_chars += len(entry) + 1
                if kept:
                    rag_section = "## Relevant Memory\n\n" + "\n".join(kept)
                    current_system_prompt = f"{system_prompt}\n\n{rag_section}"

            # Trim history to stay within context budget
            conversation = trim_history(
                conversation,
                budget_chars=BUDGET.history_budget_chars(),
            )

            # Get response (streaming — tokens print as they arrive)
            client.system_prompt = current_system_prompt
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
            conv_logger.log_turn("assistant", response)
            print()  # blank line between turns

            # Periodic soul patch check — runs in background, never blocks
            message_count += 1
            if message_count % SOUL_UPDATE_EVERY == 0:
                maybe_update_soul(soul, conversation, MODEL, OLLAMA_BASE_URL)
            if message_count % SOUL_CURIOSITY_EVERY == 0:
                maybe_grow_curiosity(soul, conversation, MODEL, OLLAMA_BASE_URL)
    finally:
        _save_session_summary(conversation, memory_store)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
