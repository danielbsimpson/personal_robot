"""
Streamlit chat UI for the personal robot LLM.

Run with:
    streamlit run src/app.py

Talks to a locally running Ollama server. Reuses OllamaClient from
src/llm/client.py — no API logic is duplicated here.
"""

import base64
import json
import sys
import os
from collections.abc import Generator
from pathlib import Path

import requests
import streamlit as st

# Allow `src.*` imports when launched as `streamlit run src/app.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.client import OllamaClient, trim_history, OLLAMA_BASE_URL, DEFAULT_MODEL
from src.llm.context import ContextBudget
from src.llm.prompts import BASE_SYSTEM_PROMPT, RESPONSE_CONSTRAINT, get_time_section
from src.memory.soul import SoulFile, maybe_update_soul, maybe_grow_curiosity, SOUL_UPDATE_EVERY, SOUL_CURIOSITY_EVERY
from src.memory.policy import is_filler_message
from src.memory.extractor import maybe_extract_memories, EXTRACT_EVERY
from src.memory.vector_store import MemoryStore
from src.memory.summariser import summarise_session
from src.utils.log import ConversationLogger, get_logger, _DEFAULT_LOGS_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAGE_TITLE = "Orion"
BUDGET = ContextBudget()  # defaults: 8192 tokens, 512 reserve
_LOGS_DIR = _DEFAULT_LOGS_DIR

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BG_PATH = _REPO_ROOT / "images" / "background_space.jpg"
_BG_B64 = base64.b64encode(_BG_PATH.read_bytes()).decode() if _BG_PATH.exists() else ""

# Ensure the memory log stub exists so the tab is ready for Phase 2.
get_logger("memory")


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

st.set_page_config(page_title=PAGE_TITLE, page_icon="⭐", layout="wide")

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Hide Streamlit default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Subtle sidebar right border */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(74, 144, 217, 0.25);
    }

    /* Orion identity pill used in sidebar header */
    .orion-pill {
        display: inline-block;
        background: linear-gradient(90deg, #1a3a5c 0%, #2d6a9f 100%);
        color: #e8f4fd !important;
        padding: 3px 14px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 0 0 6px;
    }

    /* Empty/welcome screen */
    .orion-welcome {
        text-align: center;
        padding: 60px 20px;
        opacity: 0.7;
    }
    .orion-welcome .star { font-size: 3.5rem; line-height: 1; }
    .orion-welcome h3 { margin: 14px 0 6px; font-size: 1.4rem; color: #e8f4fd; }
    .orion-welcome p { font-size: 0.9rem; color: #b0cce8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Inject background image into the main content area (done separately so we can
# use the runtime-computed base64 string — f-strings can't go inside triple-quote
# CSS blocks cleanly when the string itself may be empty).
if _BG_B64:
    st.markdown(
        f"""
        <style>
        /* Background image for the main chat area only (not the sidebar) */
        .stMain {{
            background-image: url("data:image/jpeg;base64,{_BG_B64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        /* Darken chat message bubbles slightly for contrast against the space bg */
        [data-testid="stChatMessage"] {{
            background: rgba(10, 20, 40, 0.65);
            border-radius: 10px;
            backdrop-filter: blur(4px);
        }}
        /* Title + caption area overlay */
        .stMain h1, .stMain p, .stMain .stCaption {{
            text-shadow: 0 1px 6px rgba(0,0,0,0.8);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "conversation" not in st.session_state:
    st.session_state.conversation: list[dict] = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt: str = BASE_SYSTEM_PROMPT

if "selected_model" not in st.session_state:
    st.session_state.selected_model: str = DEFAULT_MODEL

if "context_trimmed" not in st.session_state:
    st.session_state.context_trimmed: bool = False

if "message_count" not in st.session_state:
    st.session_state.message_count: int = 0

if "conv_logger" not in st.session_state:
    st.session_state.conv_logger = ConversationLogger(model=st.session_state.selected_model)

if "memory_store" not in st.session_state:
    st.session_state.memory_store = MemoryStore()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p class="orion-pill">⭐&nbsp;&nbsp;Orion</p>', unsafe_allow_html=True)
    st.subheader("⚙️ Settings")

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
    if st.button("🗑️ Clear conversation", use_container_width=True):        # Persist a summary from the departing session before wiping history.
        if st.session_state.conversation:
            _summary = summarise_session(
                st.session_state.conversation,
                model=st.session_state.selected_model,
                base_url=OLLAMA_BASE_URL,
            )
            if _summary:
                st.session_state.memory_store.add_memory(
                    _summary,
                    {"source": "session_summary"},
                )
        st.session_state.conversation = []
        st.session_state.message_count = 0

    st.divider()

    # --- Soul file viewer ---
    with st.expander("🔮 Soul file", expanded=False):
        soul_view = SoulFile()
        st.code(soul_view.as_yaml_string(), language="yaml")
        st.caption("Updated automatically during conversation. You can also edit `data/soul.yaml` directly.")

    # --- Manual soul update trigger ---
    if st.button("⚡ Force soul update now", use_container_width=True):
        soul_force = SoulFile()
        if st.session_state.conversation:
            maybe_update_soul(
                soul_force,
                st.session_state.conversation,
                st.session_state.selected_model,
                OLLAMA_BASE_URL,
            )
            maybe_grow_curiosity(
                soul_force,
                st.session_state.conversation,
                st.session_state.selected_model,
                OLLAMA_BASE_URL,
            )
            st.toast("⚡ Soul update triggered — check the log in a few seconds", icon="⚡")
        else:
            st.warning("No conversation to analyse yet.")

    st.divider()

    # --- Worker log viewer (tabbed) ---
    with st.expander("📋 Logs", expanded=False):
        tab_conv, tab_soul, tab_trim, tab_mem = st.tabs(
            ["Conversations", "Soul changes", "Context trim", "Memory"]
        )

        def _last_lines(path, n: int = 50) -> str:
            if not path.exists():
                return ""
            lines = path.read_text(encoding="utf-8").splitlines()
            return "\n".join(lines[-n:]) if lines else ""

        with tab_conv:
            conv_dir = _LOGS_DIR / "conversations"
            files = sorted(conv_dir.glob("*.jsonl")) if conv_dir.exists() else []
            if files:
                latest = files[-1]
                st.caption(f"`{latest.name}`")
                st.code(_last_lines(latest), language="json")
                if st.button("Clear", key="clear_conv_log"):
                    latest.write_text("", encoding="utf-8")
                    st.rerun()
            else:
                st.caption("No conversation log yet.")

        with tab_soul:
            soul_log = _LOGS_DIR / "soul_changes.log"
            content = _last_lines(soul_log)
            if content:
                st.code(content, language="text")
                if st.button("Clear", key="clear_soul_log"):
                    soul_log.write_text("", encoding="utf-8")
                    st.rerun()
            else:
                st.caption("No soul changes logged yet.")

        with tab_trim:
            trim_log = _LOGS_DIR / "context_trim.log"
            content = _last_lines(trim_log)
            if content:
                st.code(content, language="text")
                if st.button("Clear", key="clear_trim_log"):
                    trim_log.write_text("", encoding="utf-8")
                    st.rerun()
            else:
                st.caption("No context trims logged yet.")

        with tab_mem:
            mem_log = _LOGS_DIR / "memory.log"
            content = _last_lines(mem_log)
            if content:
                st.code(content, language="text")
                if st.button("Clear", key="clear_mem_log"):
                    mem_log.write_text("", encoding="utf-8")
                    st.rerun()
            else:
                st.caption("No memory log yet — wired in Phase 2.")

    st.caption(f"Model: `{st.session_state.selected_model}`")
    st.caption(f"History: {len(st.session_state.conversation)} messages")
    if st.session_state.get("context_trimmed"):
        st.markdown(
            """
            <a
                href="javascript:void(0)"
                onclick="(function(){
                    var details = Array.from(document.querySelectorAll('details'));
                    var logsDetails = details.find(function(d){
                        var s = d.querySelector('summary');
                        return s && s.innerText.indexOf('Logs') !== -1;
                    });
                    if (!logsDetails) return;
                    if (!logsDetails.open) logsDetails.querySelector('summary').click();
                    setTimeout(function(){
                        var tabs = Array.from(document.querySelectorAll('[role=tab]'));
                        var trimTab = tabs.find(function(t){
                            return t.innerText.trim() === 'Context trim';
                        });
                        if (trimTab) {
                            trimTab.click();
                            trimTab.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                        }
                    }, 300);
                })()"
                style="display:block;padding:8px 10px;margin:4px 0;background:#fff3cd;
                       border:1px solid #ffc107;border-radius:6px;color:#856404;
                       text-decoration:none;font-size:0.85rem;font-weight:500;
                       cursor:pointer;"
            >&#9888;&#65039; Context trimmed &mdash; click to view trim log</a>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.markdown("# ⭐ Orion")
st.caption("Your personal AI — running fully locally on your machine")
st.divider()

# Render existing conversation history
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Welcome screen when no conversation has started yet
if not st.session_state.conversation:
    st.markdown(
        """
        <div class="orion-welcome">
            <div class="star">⭐</div>
            <h3>Hello, I'm Orion</h3>
            <p>Start a conversation below.<br>I'm running fully locally — no internet connection required.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Chat input
user_input = st.chat_input("Message Orion…")

if user_input:
    # Build the client — compose system prompt with soul section and current time appended
    soul = SoulFile()
    soul_section = soul.to_prompt_section(budget_chars=BUDGET.soul_budget_chars())
    combined_prompt = st.session_state.system_prompt
    if soul_section:
        combined_prompt = f"{st.session_state.system_prompt}\n\n{soul_section}"
    time_section = get_time_section()
    if time_section:
        combined_prompt = f"{combined_prompt}\n\n{time_section}"

    # RAG injection — query long-term memory for context relevant to this message
    if not is_filler_message(user_input):
        rag_results = st.session_state.memory_store.query_memory(user_input)
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
                combined_prompt = f"{combined_prompt}\n\n{rag_section}"

    # Always append the output constraint last so it is closest to the conversation
    combined_prompt = f"{combined_prompt}\n\n{RESPONSE_CONSTRAINT}"

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

    # Relevance pre-filter: filler messages ("ok", "thanks", etc.) are shown
    # but excluded from the history buffer so they don't waste context budget.
    _is_filler = is_filler_message(user_input)

    if not _is_filler:
        st.session_state.conversation.append({"role": "user", "content": user_input})
    st.session_state.conv_logger.log_turn("user", user_input, model=st.session_state.selected_model)

    # Trim history to stay within context budget before sending.
    # For filler turns, pass the history without the new user message so the
    # LLM still generates a reply with full context.
    trimmed = trim_history(
        list(st.session_state.conversation),
        budget_chars=BUDGET.history_budget_chars(),
    )
    # If it was filler, append the message only for this one LLM call.
    history_for_llm = (
        trimmed + [{"role": "user", "content": user_input}]
        if _is_filler
        else trimmed
    )
    st.session_state.context_trimmed = len(trimmed) < len(st.session_state.conversation)

    # Stream the assistant response into a chat bubble
    with st.chat_message("assistant"):
        response_text = st.write_stream(token_stream(client, history_for_llm))

    # Persist the full response into session state (skip filler exchanges)
    if not _is_filler:
        st.session_state.conversation.append(
            {"role": "assistant", "content": response_text}
        )
    st.session_state.conv_logger.log_turn(
        "assistant", response_text, model=st.session_state.selected_model
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
    if st.session_state.message_count % EXTRACT_EVERY == 0 and not _is_filler:
        maybe_extract_memories(
            soul,
            st.session_state.conversation,
            st.session_state.selected_model,
            OLLAMA_BASE_URL,
            vector_store=st.session_state.memory_store,
        )
