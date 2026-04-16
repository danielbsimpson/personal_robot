"""
Streamlit chat UI for the personal robot LLM.

Run with:
    streamlit run src/app.py

Talks to a locally running Ollama server. Reuses OllamaClient from
src/llm/client.py — no API logic is duplicated here.
"""

import json
import sys
import os
from collections.abc import Generator

import requests
import streamlit as st

# Allow `src.*` imports when launched as `streamlit run src/app.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.client import OllamaClient, trim_history, OLLAMA_BASE_URL, DEFAULT_MODEL
from src.llm.prompts import BASE_SYSTEM_PROMPT
from src.memory.soul import SoulFile, maybe_update_soul, maybe_grow_curiosity, SOUL_UPDATE_EVERY, SOUL_CURIOSITY_EVERY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAGE_TITLE = "Personal Robot Chat"
CONTEXT_LIMIT_CHARS = 4096 * 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def fetch_available_models(base_url: str) -> list[str]:
    """Query Ollama /api/tags and return a sorted list of model names.

    Cached for 30 seconds so repeated reruns don't hit the network.
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return sorted(models) if models else [DEFAULT_MODEL]
    except requests.RequestException:
        return [DEFAULT_MODEL]


@st.cache_data(ttl=10)
def check_ollama_available(base_url: str) -> bool:
    """Return True if the Ollama server is reachable.

    Cached for 10 seconds to avoid an HTTP round-trip on every rerender.
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def token_stream(client: OllamaClient, messages: list[dict]) -> Generator[str, None, None]:
    """Yield response tokens one at a time for use with st.write_stream."""
    payload = client._build_payload(messages, stream=True)
    url = f"{client.base_url}/api/chat"
    with client._session.post(url, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
            if chunk.get("done"):
                break


# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(page_title=PAGE_TITLE, page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "conversation" not in st.session_state:
    st.session_state.conversation: list[dict] = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt: str = BASE_SYSTEM_PROMPT

if "selected_model" not in st.session_state:
    st.session_state.selected_model: str = DEFAULT_MODEL

if "message_count" not in st.session_state:
    st.session_state.message_count: int = 0

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    # --- Ollama status badge ---
    if check_ollama_available(OLLAMA_BASE_URL):
        st.success("Ollama online", icon="✅")
    else:
        st.error("Ollama offline — start Ollama and refresh", icon="🔴")

    st.divider()

    # --- Model selector ---
    # key="selected_model" binds directly to st.session_state — no manual
    # st.rerun() needed; the widget change itself triggers exactly one rerun.
    available_models = fetch_available_models(OLLAMA_BASE_URL)
    if st.session_state.selected_model not in available_models:
        st.session_state.selected_model = available_models[0]
    st.selectbox(
        "Model",
        options=available_models,
        key="selected_model",
        help="Models currently pulled in Ollama",
    )

    st.divider()

    # --- System prompt editor ---
    st.subheader("System Prompt")
    new_prompt = st.text_area(
        label="system_prompt",
        value=st.session_state.system_prompt,
        height=220,
        label_visibility="collapsed",
        help="Edit the system prompt. Changes take effect on the next message.",
    )
    if new_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = new_prompt

    st.divider()

    # --- Clear conversation ---
    # Button click triggers a natural rerun; no explicit st.rerun() needed.
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.conversation = []
        st.session_state.message_count = 0

    st.divider()

    # --- Soul file viewer ---
    with st.expander("🔮 Soul file", expanded=False):
        soul_view = SoulFile()
        st.code(soul_view.as_yaml_string(), language="yaml")
        st.caption("Updated automatically during conversation. You can also edit `data/soul.yaml` directly.")

    st.caption(f"Model: `{st.session_state.selected_model}`")
    st.caption(f"History: {len(st.session_state.conversation)} messages")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("🤖 Personal Robot Chat")

# Render existing conversation history
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type a message…")

if user_input:
    # Build the client — compose system prompt with soul section appended
    soul = SoulFile()
    soul_section = soul.to_prompt_section()
    combined_prompt = st.session_state.system_prompt
    if soul_section:
        combined_prompt = f"{st.session_state.system_prompt}\n\n{soul_section}"

    client = OllamaClient(
        model=st.session_state.selected_model,
        system_prompt=combined_prompt,
    )

    if not check_ollama_available(OLLAMA_BASE_URL):
        st.error("Ollama is not reachable. Make sure `ollama serve` is running.")
        st.stop()

    # Show and record the user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.conversation.append({"role": "user", "content": user_input})

    # Trim history to stay within context budget before sending
    trimmed = trim_history(
        list(st.session_state.conversation), limit_chars=CONTEXT_LIMIT_CHARS
    )

    # Stream the assistant response into a chat bubble
    with st.chat_message("assistant"):
        response_text = st.write_stream(token_stream(client, trimmed))

    # Persist the full response into session state
    st.session_state.conversation.append(
        {"role": "assistant", "content": response_text}
    )

    # Periodic soul patch check — background thread, never blocks the UI
    st.session_state.message_count += 1
    if st.session_state.message_count % SOUL_UPDATE_EVERY == 0:
        maybe_update_soul(
            soul,
            st.session_state.conversation,
            st.session_state.selected_model,
            OLLAMA_BASE_URL,
        )
    if st.session_state.message_count % SOUL_CURIOSITY_EVERY == 0:
        maybe_grow_curiosity(
            soul,
            st.session_state.conversation,
            st.session_state.selected_model,
            OLLAMA_BASE_URL,
        )
