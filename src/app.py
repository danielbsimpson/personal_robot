"""
Streamlit chat UI for the personal robot LLM.

Run with:
    streamlit run src/app.py

Talks to a locally running Ollama server. Reuses OllamaClient from
src/llm/client.py — no API logic is duplicated here.
"""

import base64
import json
import random
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
from src.llm.prompts import BASE_SYSTEM_PROMPT, CURIOSITY_NUDGE, RESPONSE_CONSTRAINT, RESPONSE_REMINDER, get_time_section
from src.memory.soul import SoulFile, maybe_update_soul, maybe_grow_curiosity, migrate_soul_to_facts, migrate_soul_to_claims, SOUL_UPDATE_EVERY, SOUL_CURIOSITY_EVERY
from src.memory.policy import is_filler_message
from src.memory.extractor import maybe_extract_memories, EXTRACT_EVERY
from src.memory.vector_store import MemoryStore
from src.memory.facts_store import FactsStore
from src.memory.claims import ClaimsStore
from src.memory.consolidation import ConsolidationEngine
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
    /* Tighten top padding and reserve space for the fixed chat input */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
    }

    /* Pin the chat input to the bottom of the viewport when rendered inside a
       fragment (Streamlit only does this automatically at the top level).
       The backdrop keeps it legible over the space background image. */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 999 !important;
        padding: 0.5rem 1rem 1rem !important;
        background: rgba(14, 17, 23, 0.85) !important;
        backdrop-filter: blur(8px) !important;
    }

    /* Hide Streamlit default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Subtle sidebar right border — removed (no sidebar) */

    /* Orion identity pill — removed (no sidebar) */

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

if "curiosity_counter" not in st.session_state:
    st.session_state.curiosity_counter: int = 0

if "curiosity_threshold" not in st.session_state:
    st.session_state.curiosity_threshold: int = random.randint(4, 10)

if "conv_logger" not in st.session_state:
    st.session_state.conv_logger = ConversationLogger(model=st.session_state.selected_model)

if "memory_store" not in st.session_state:
    st.session_state.memory_store = MemoryStore()

if "facts_store" not in st.session_state:
    st.session_state.facts_store = FactsStore()

if "claims_store" not in st.session_state:
    st.session_state.claims_store = ClaimsStore()

# Run soul → facts and soul → claims migration once per session
if "soul_migrated" not in st.session_state:
    _migration_soul = SoulFile()
    migrate_soul_to_facts(_migration_soul)
    migrate_soul_to_claims(_migration_soul)
    st.session_state.soul_migrated = True

# ---------------------------------------------------------------------------
# Log helper
# ---------------------------------------------------------------------------

def _last_lines(path, n: int = 50) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-n:]) if lines else ""


# ---------------------------------------------------------------------------
# Main chat area — wrapped in a fragment so only this section reruns on each
# message, leaving the sidebar and page chrome completely undisturbed.
# ---------------------------------------------------------------------------

@st.fragment
def chat_area() -> None:
    st.markdown("# ⭐ Orion")
    st.caption("Your personal AI — running fully locally on your machine")
    st.divider()

    # Render existing conversation history
    for msg in st.session_state.conversation:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input lives inside the fragment so submitting triggers only a fragment
    # rerun — not a full page reload. CSS above pins it visually to the bottom.
    user_input = st.chat_input("Message Orion…", key="main_chat_input")

    # Welcome screen — hidden while a message is in flight to avoid a flash
    if not st.session_state.conversation and not user_input:
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

    if user_input:
        _is_filler = is_filler_message(user_input)

        # Increment curiosity counter for non-filler messages and decide whether
        # Orion should ask a curiosity question this turn.
        if not _is_filler:
            st.session_state.curiosity_counter += 1
        _soul_data = SoulFile().load()
        _curiosity_queue = (_soul_data.get("identity") or {}).get("curiosity_queue") or []
        _curiosity_active = (
            not _is_filler
            and bool(_curiosity_queue)
            and st.session_state.curiosity_counter >= st.session_state.curiosity_threshold
        )

        # Show and record the user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Scroll to bottom as soon as the user message is visible so it doesn't
        # sit off-screen while Orion is thinking.
        st.html(
            """
            <script>
            (function() {
                var el = window.parent.document.querySelector('[data-testid="stMain"]');
                if (!el) el = window.parent.document.querySelector('section.main');
                if (el) el.scrollTop = el.scrollHeight;
            })();
            </script>
            """
        )

        # Relevance pre-filter: filler messages ("ok", "thanks", etc.) are shown
        # but excluded from the history buffer so they don't waste context budget.
        if not _is_filler:
            st.session_state.conversation.append({"role": "user", "content": user_input})
        st.session_state.conv_logger.log_turn("user", user_input, model=st.session_state.selected_model)

        with st.chat_message("assistant"):
            # Show a spinner while doing all the heavy pre-processing (soul load,
            # RAG/facts queries, prompt assembly) before the first token arrives.
            with st.spinner("Thinking…"):
                if not check_ollama_available(OLLAMA_BASE_URL):
                    st.error("Ollama is not reachable. Make sure `ollama serve` is running.")
                    st.stop()

                # Build the client — compose system prompt with soul section and current time appended
                soul = SoulFile()
                soul_section = soul.to_prompt_section(budget_chars=BUDGET.soul_budget_chars())
                combined_prompt = st.session_state.system_prompt
                if soul_section:
                    combined_prompt = f"{st.session_state.system_prompt}\n\n{soul_section}"
                time_section = get_time_section()
                if time_section:
                    combined_prompt = f"{combined_prompt}\n\n{time_section}"

                # Context injection: claims → facts → episodes (claims first = highest trust)
                if not _is_filler:
                    # Claims injection — trust-calibrated long-term knowledge
                    claims_results = st.session_state.claims_store.query_claims(
                        user_input, n_results=8
                    )
                    if claims_results:
                        claims_budget = BUDGET.claims_budget_chars()
                        kept_claims, total_chars = [], 0
                        for c in claims_results:
                            entry = f"- {c['text']}"
                            if total_chars + len(entry) + 1 > claims_budget:
                                break
                            kept_claims.append(entry)
                            total_chars += len(entry) + 1
                        if kept_claims:
                            claims_section = "## Long-Term Knowledge\n\n" + "\n".join(kept_claims)
                            combined_prompt = f"{combined_prompt}\n\n{claims_section}"

                    # Facts store injection — structured facts retrieved by keyword/category
                    facts_results = st.session_state.facts_store.query_facts(user_input)
                    if facts_results:
                        facts_budget = BUDGET.facts_budget_chars()
                        kept_facts, total_chars = [], 0
                        for fact in facts_results:
                            entry = f"- {fact}"
                            if total_chars + len(entry) + 1 > facts_budget:
                                break
                            kept_facts.append(entry)
                            total_chars += len(entry) + 1
                        if kept_facts:
                            facts_section = "## Relevant Facts\n\n" + "\n".join(kept_facts)
                            combined_prompt = f"{combined_prompt}\n\n{facts_section}"

                    # RAG injection — episodic session summaries
                    rag_results = st.session_state.memory_store.query_memory(user_input)
                    if rag_results:
                        rag_budget = BUDGET.episodes_budget_chars()
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

                # Inject curiosity nudge when the counter has reached the threshold
                if _curiosity_active:
                    combined_prompt = f"{combined_prompt}\n\n{CURIOSITY_NUDGE}"

                # Always append the output constraint last so it is closest to the conversation
                combined_prompt = f"{combined_prompt}\n\n{RESPONSE_CONSTRAINT}"

                client = OllamaClient(
                    model=st.session_state.selected_model,
                    system_prompt=combined_prompt,
                )

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

                # Inject a terse reminder as the last system message immediately
                # before the user turn — closest possible position to generation.
                if history_for_llm and history_for_llm[-1]["role"] == "user":
                    history_for_llm = (
                        history_for_llm[:-1]
                        + [{"role": "system", "content": RESPONSE_REMINDER}]
                        + [history_for_llm[-1]]
                    )

            # Spinner exits here — stream tokens directly into the chat bubble
            response_text = st.write_stream(token_stream(client, history_for_llm))

        # Persist the full response into session state (skip filler exchanges)
        if not _is_filler:
            st.session_state.conversation.append(
                {"role": "assistant", "content": response_text}
            )
        st.session_state.conv_logger.log_turn(
            "assistant", response_text, model=st.session_state.selected_model
        )

        # Reset curiosity counter and re-roll threshold after Orion asks a question
        if _curiosity_active:
            st.session_state.curiosity_counter = 0
            st.session_state.curiosity_threshold = random.randint(4, 10)

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
                facts_store=st.session_state.facts_store,
            )

    # Scroll the chat area to the bottom after every fragment rerun so the
    # latest message is always visible without manual scrolling.
    st.html(
        """
        <script>
        (function() {
            var el = window.parent.document.querySelector('[data-testid="stMain"]');
            if (!el) el = window.parent.document.querySelector('section.main');
            if (el) el.scrollTop = el.scrollHeight;
        })();
        </script>
        """
    )


tab_chat, tab_settings = st.tabs(["💬 Chat", "⚙️ Settings"])

with tab_chat:
    chat_area()

with tab_settings:
    st.subheader("⚙️ Settings")

    col_status, col_model = st.columns([1, 2])
    with col_status:
        if check_ollama_available(OLLAMA_BASE_URL):
            st.success("Ollama online", icon="✅")
        else:
            st.error("Ollama offline", icon="🔴")
    with col_model:
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

    col_clear, col_soul = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear conversation", use_container_width=True):
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
                # Trigger consolidation in the background after saving the summary
                _engine = ConsolidationEngine(
                    claims_store=st.session_state.claims_store,
                    memory_store=st.session_state.memory_store,
                    model=st.session_state.selected_model,
                    base_url=OLLAMA_BASE_URL,
                )
                _engine.run_async()
            st.session_state.conversation = []
            st.session_state.message_count = 0
    with col_soul:
        if st.button("⚡ Force soul update", use_container_width=True):
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

    with st.expander("🔮 Soul file", expanded=False):
        soul_view = SoulFile()
        st.code(soul_view.as_yaml_string(), language="yaml")
        st.caption("Updated automatically during conversation. You can also edit `data/soul.yaml` directly.")

    st.divider()

    with st.expander("🧠 Memory health", expanded=False):
        try:
            _counts = st.session_state.claims_store.count()
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Claims", _counts["total"])
            col_b.metric("Stale (30d)", _counts["stale"])
            col_c.metric("Contradicted", _counts["contradicted"])
        except Exception:
            st.caption("Claims store unavailable.")
        if st.button("⚡ Run consolidation now", use_container_width=True):
            _eng = ConsolidationEngine(
                claims_store=st.session_state.claims_store,
                memory_store=st.session_state.memory_store,
                model=st.session_state.selected_model,
                base_url=OLLAMA_BASE_URL,
            )
            _eng.run_async()
            st.toast("Consolidation triggered — new claims will appear shortly.", icon="🧠")

    st.divider()

    st.subheader("📋 Logs")
    if st.session_state.get("context_trimmed"):
        st.warning("Context was trimmed this session — see the Context trim tab below.", icon="⚠️")

    log_conv, log_soul_tab, log_trim, log_mem = st.tabs(
        ["Conversations", "Soul changes", "Context trim", "Memory"]
    )
    with log_conv:
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
    with log_soul_tab:
        soul_log = _LOGS_DIR / "soul_changes.log"
        content = _last_lines(soul_log)
        if content:
            st.code(content, language="text")
            if st.button("Clear", key="clear_soul_log"):
                soul_log.write_text("", encoding="utf-8")
                st.rerun()
        else:
            st.caption("No soul changes logged yet.")
    with log_trim:
        trim_log = _LOGS_DIR / "context_trim.log"
        content = _last_lines(trim_log)
        if content:
            st.code(content, language="text")
            if st.button("Clear", key="clear_trim_log"):
                trim_log.write_text("", encoding="utf-8")
                st.rerun()
        else:
            st.caption("No context trims logged yet.")
    with log_mem:
        mem_log = _LOGS_DIR / "memory.log"
        content = _last_lines(mem_log)
        if content:
            st.code(content, language="text")
            if st.button("Clear", key="clear_mem_log"):
                mem_log.write_text("", encoding="utf-8")
                st.rerun()
        else:
            st.caption("No memory log yet.")

    st.divider()
    st.caption(f"Model: `{st.session_state.selected_model}`")
    st.caption(f"History: {len(st.session_state.conversation)} messages")

