# TODO — Personal Robot Build Plan

All tasks are grouped by phase. Complete each phase before moving to the next, as later phases depend on earlier ones.

Legend: `[ ]` = not started · `[~]` = in progress · `[x]` = done

---

## Phase 0 — Environment Setup

- [x] Install [Ollama for Windows](https://ollama.com/download) — v0.20.7
- [x] Install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) and verify GPU is detected (`nvidia-smi`) — RTX 4060 Laptop GPU confirmed, driver 596.21
- [x] Install Python 3.11 and confirm version (`python --version`) — 3.11.9
- [x] Create project virtual environment: `python -m venv .venv`
- [x] Create `requirements.txt` with initial dependencies (see Phase 1)
- [x] Create `.gitignore` to exclude `data/`, `models/`, `.venv/`, `__pycache__/`
- [x] Create base project folder structure (`src/llm`, `src/memory`, `src/audio`, `src/vision`, `src/robot`, `data/memory`, `models/`, `tests/`)
- [x] Install [FFmpeg for Windows](https://ffmpeg.org/download.html) and add to PATH (required by audio tools) — v8.1

---

## Phase 1 — Local LLM

**Goal**: A Python script that sends a prompt to a locally running LLM and receives a response. No internet required.

### 1.1 — Install and Verify Ollama

- [x] Pull `phi4-mini` model via Ollama: `ollama pull phi4-mini` — 2.5 GB, Q4_K_M
- [x] Verify Ollama is using the GPU — RTX 4060 VRAM spiked +3,403 MiB during inference, confirming GPU usage
- [x] Confirm the Ollama local API is accessible at `http://localhost:11434`

### 1.2 — LLM Client Wrapper (`src/llm/client.py`)

- [x] Write an `OllamaClient` class that wraps the Ollama HTTP API (`/api/chat` endpoint)
- [x] Implement a `chat(messages: list[dict]) -> str` method that sends a messages array and returns the assistant response
- [x] Implement streaming response support (print tokens as they arrive)
- [x] Add a configurable `system_prompt` parameter to `OllamaClient`
- [x] Write `src/llm/prompts.py` with a base system prompt defining the robot's persona and behavioural guidelines
- [x] Write a basic test in `tests/test_llm.py` to confirm a round-trip prompt/response works — 7/7 passed

### 1.3 — Conversation Loop (`src/main.py`)

- [x] Build a simple CLI conversation loop (type → LLM response → type again)
- [x] Maintain a rolling conversation history (list of `{"role": ..., "content": ...}` dicts)
- [x] Add a configurable context window limit (trim oldest messages when history exceeds N tokens)
- [x] Add graceful exit on `quit` / `exit` / `Ctrl+C`

---

## Phase 1.5 — Streamlit Chat UI

**Goal**: A local web app that provides a chat interface to the LLM, making it easy to test multi-turn conversations without the CLI.

### 1.5.1 — Install Dependencies

- [x] Add `streamlit>=1.40` to `requirements.txt`
- [x] Install into venv: `pip install streamlit`

### 1.5.2 — Build the Chat App (`src/app.py`)

- [x] Create `src/app.py` using `st.chat_message` / `st.chat_input` for the conversation UI
- [x] Persist conversation history in `st.session_state` so it survives re-renders
- [x] Reuse `OllamaClient` from `src/llm/client.py` — no duplication of API logic
- [x] Stream tokens into the chat bubble using `st.write_stream` for a live typing effect
- [x] Add a sidebar with:
  - Model selector (populated by querying `GET /api/tags` from Ollama)
  - System prompt text area (editable at runtime, defaults to `BASE_SYSTEM_PROMPT`)
  - "Clear conversation" button that resets `st.session_state`
- [x] Show a status badge ("Ollama online / offline") based on `client.is_available()`
- [x] Handle Ollama being unavailable with a user-friendly error message

### 1.5.3 — Smoke Test

- [x] Launch with `streamlit run src/app.py` and verify the app opens in the browser
- [x] Send at least 5 messages in a row and confirm history is maintained correctly
- [x] Test the "Clear conversation" button resets the chat
- [x] Verify the model selector switches models mid-session without error — tested with `phi4-mini` → `llama3.2:3b`

---

## Phase 1.7 — Soul File (Identity & Long-Term Facts)

**Goal**: Give the LLM a persistent YAML file that defines its identity, stores facts about the user, and records knowledge about its environment. The LLM can propose updates to this file during conversation, so it learns and grows over time without requiring the full RAG stack.

This is distinct from Phase 2 (episodic RAG memory):
- **Soul file** = *who am I / who is the user / what do I know* — structured, human-readable, hand-editable
- **RAG memory** = *what happened in past conversations* — vector search over session summaries

### 1.7.1 — Soul File Format (`data/soul.yaml`)

- [x] Define a YAML schema with the following top-level sections:
  ```yaml
  identity:        # Robot's name, personality traits, capabilities
  user:            # User's name, preferences, facts learned about them
  environment:     # Location, connected hardware, known surroundings
  facts:           # General facts the robot has learned and wants to retain
  ```
- [x] Create a default `data/soul.yaml` seeded with the robot's base identity (name: Orion, hardware, capabilities) and user name (Daniel Bowman Simpson)
- [x] Add `data/soul.yaml` to `.gitignore` so personal facts are not committed — already covered by the `data/` rule

### 1.7.2 — Soul File Module (`src/memory/soul.py`)

- [x] Create a `SoulFile` class that reads and writes `data/soul.yaml`
- [x] Implement `load() -> dict` — parses YAML and returns the full soul dict
- [x] Implement `to_prompt_section() -> str` — formats the soul as a `## About Me` YAML block for injection into the system prompt
- [x] Implement `update(section: str, key: str, value: str)` — writes a single fact to the correct section and saves the file
- [x] Implement `apply_patch(patch: dict)` — bulk-applies a dict of `{section: {key: value}}` updates from the LLM
- [x] Add file locking via a module-level `threading.Lock` + atomic temp-file rename so concurrent writes never corrupt the file

### 1.7.3 — LLM-Driven Updates

- [x] Add `SOUL_PATCH_PROMPT` to `src/llm/prompts.py` — instructs the LLM to output a fenced JSON patch block if new facts were learned, or nothing at all
- [x] `_extract_json_patch(text)` helper parses the first ` ```json ... ``` ` block from the LLM response
- [x] `_patch_worker` daemon thread: calls a fresh `OllamaClient`, gets patch, calls `soul.apply_patch()` if valid
- [x] `maybe_update_soul()` spawns the daemon thread — fires every `SOUL_UPDATE_EVERY = 3` user messages

### 1.7.4 — Integrate Soul File into Conversation Loop

- [x] `src/main.py` — loads soul on startup, injects `soul.to_prompt_section()` into system prompt, fires `maybe_update_soul` every 3 turns
- [x] `src/app.py` — same soul injection per message, soul section appended to the user-editable system prompt; `message_count` tracked in `st.session_state`
- [x] Added a "🔮 Soul file" expander in the Streamlit sidebar showing live YAML contents (read-only view, re-reads file on each render to reflect background updates)
- [x] Write `tests/test_soul.py` — 21/21 passed (load, save, update, apply_patch, to_prompt_section, as_yaml_string, _extract_json_patch)
- [x] **Soul file scope**: the soul file is the *always-on identity layer* — it should stay compact and contain only stable, perpetually-relevant facts (Orion's identity and Daniel's core profile). Episodic facts learned from individual conversations belong in Phase 2 RAG, not accumulated in the soul file's `facts` section.

---

## Phase 1.8 — Logging & Observability

**Goal**: A unified logging layer that makes every significant runtime event reviewable after the fact — soul file changes, what was trimmed from the context window, what the LLM was actually sent, and (in Phase 2) what long-term memories were retrieved or written.

### 1.8.1 — Central Logging Setup (`src/utils/log.py`)

- [x] Create `src/utils/__init__.py` and `src/utils/log.py`
- [x] Implement `get_logger(name: str) -> logging.Logger` — returns a logger that writes to `data/logs/<name>.log` (rotating, max 5 MB × 3 backups) and WARNING+ to console
- [x] All existing ad-hoc loggers in `soul.py` (`soul_worker`, `curiosity_worker`) migrated to use `get_logger()`
- [x] Ensure `data/logs/` is added to `.gitignore` (alongside existing `data/` rule — confirm coverage)

### 1.8.2 — Conversation Turn Log

- [x] On each completed turn (user message + assistant response), append a JSON object to `data/logs/conversations/<YYYY-MM-DD_HH-MM-SS>.jsonl`:
  ```json
  {"ts": "2026-04-16T14:32:01", "role": "user", "content": "...", "model": "phi4-mini"}
  {"ts": "2026-04-16T14:32:04", "role": "assistant", "content": "...", "model": "phi4-mini"}
  ```
- [x] A new `.jsonl` file is created per session (keyed by session start time) so sessions are naturally separated
- [x] Implement in both `src/main.py` (CLI) and `src/app.py` (Streamlit) using a shared `ConversationLogger` class in `src/utils/log.py`

### 1.8.3 — Soul File Audit Log

- [x] Replace the unstructured `soul_worker.log` with a richer `data/logs/soul_changes.log`
- [x] Each patch entry records: timestamp, the patch dict that was applied, and a before/after snapshot of the changed keys (not the full file — just the modified paths)
- [x] Each "no patch" event is recorded as a single INFO line (timestamp + "no new facts")
- [x] Migrate all existing `_log.info` / `_log.debug` calls in `_patch_worker` and `_curiosity_worker` to the new logger

### 1.8.4 — Short-Term Memory (Context Trim) Log

- [x] In `trim_history()` (`src/llm/client.py`), log to `data/logs/context_trim.log` whenever messages are dropped:
  - Number of messages before and after trim
  - Content of each dropped message (role + first 120 chars of content)
  - Total character count before and after
- [x] Only log when messages are actually dropped (no-op trims are silent)

### 1.8.5 — Long-Term Memory Log (stub — filled in Phase 2)

- [x] Create `data/logs/memory.log` via `get_logger("memory")` now; leave it empty until Phase 2
- [x] Phase 2 will add: every ChromaDB query (query text, top-K results, similarity scores, whether threshold was cleared), and every session summary write (timestamp, summary text, ChromaDB document ID)

### 1.8.6 — Streamlit Log Viewer

- [x] Replace the existing single "Worker log" expander with a tabbed `st.tabs` panel covering all four log files: **Conversations**, **Soul changes**, **Context trim**, **Memory**
- [x] Each tab shows the last 50 lines of the corresponding log file with a "Clear" button
- [x] If a log file does not exist yet (e.g. Memory before Phase 2), show a `st.caption("No log yet")` placeholder

### 1.8.7 — Test Coverage (`tests/test_logging.py`)

- [x] Test `ConversationLogger`: verify a `.jsonl` file is created in `tmp_path`, that each appended turn is valid JSON, and that role/content/ts fields are present
- [x] Test soul audit log: call `apply_patch()` on a `SoulFile` backed by `tmp_path` and verify the audit log contains the expected patch entry
- [x] Test context trim log: call `trim_history()` with a history that exceeds the limit and verify the trim log records the dropped messages
- [x] All tests use `tmp_path` fixtures and mock loggers pointed at temp directories — no production log files touched
- [x] Add a `conftest.py` fixture `log_dir(tmp_path)` that patches the `data/logs/` path globally for the test session

---

## Phase 1.9 — Context Budget Management

**Goal**: Ensure the assembled system prompt never silently overflows the model's context window. Measure every section before sending, protect the response headroom, and degrade gracefully — dropping the least-important content first — when the budget is tight.

### 1.9.1 — Token Counting Utility

- [x] Add a `count_tokens(text: str) -> int` helper to `src/llm/context.py` — approximates as `len(text) // 4` (1 token ≈ 4 chars) for zero-dependency speed
- [x] Design the interface so a more accurate counter (e.g. `tiktoken`) can be swapped in later without changing call sites
- [x] Add a module-level `CHARS_PER_TOKEN = 4` constant so the approximation is easy to find and update

### 1.9.2 — ContextBudget Class (`src/llm/context.py`)

- [x] Create `src/llm/context.py` with a `ContextBudget` dataclass holding the tier allocations:
  ```
  response_reserve  = 512 tokens  (hard floor, always protected)
  system_soul_pct   = 20%  of (total - reserve)
  history_pct       = 50%  of (total - reserve)
  rag_vision_pct    = 20%  of (total - reserve)
  misc_pct          = 10%  of (total - reserve)
  ```
- [x] Implement `assemble(soul_text, history, rag_text, vision_text, time_text) -> AssembledContext` — measures each section, applies its budget, and returns both the (possibly trimmed) text for each section and a `trimmed: set[str]` indicating which sections were cut
- [x] `AssembledContext` should expose a `total_chars` property and a `was_trimmed` boolean for easy checking at the call site

### 1.9.3 — Soul File Budget Trimming

- [x] Add an optional `budget_chars: int | None = None` parameter to `SoulFile.to_prompt_section()`
- [x] If `budget_chars` is set and the full YAML exceeds it, progressively drop sections in this order until it fits:
  1. `facts`
  2. `environment`
  3. Extended `user` fields (keep `name`, `preferred_name`, `date_of_birth`; drop everything else)
  4. `identity` non-essential fields (keep `name`, `persona`, `communication_style`; drop `capabilities`, `hardware`, `curiosity_queue`)
- [x] Log which sections were dropped via the context trim logger (Phase 1.8)
- [x] Add tests in `tests/test_soul.py` covering each trimming level — basic level-1 (facts drop) covered in `test_context.py`; exhaustive per-level soul tests not yet written

### 1.9.4 — Conversation History Budget Trimming

- [x] Refactor `trim_history()` in `src/llm/client.py` to accept an explicit `budget_chars: int` rather than the current `limit_chars` constant so `ContextBudget` drives it directly
- [x] Preserve backwards compatibility: if called without the new parameter, fall back to the existing `CONTEXT_LIMIT_CHARS` default
- [x] Existing trim-logging behaviour (Phase 1.8, task 1.8.4) remains unchanged

### 1.9.5 — RAG and Vision Budget Trimming

- [x] For vision context (Phase 5): if the vision summary exceeds its budget, truncate to the first sentence; if still over, drop entirely — implemented in `ContextBudget.assemble()`
- [x] Both trim events are logged to `context_trim.log`
- [x] In Phase 2's `query_memory()` call site: pass the RAG budget from `ContextBudget` as a `max_chars` cap — reduce `n_results` until the combined result text fits (deferred to Phase 2)

### 1.9.6 — Wire ContextBudget into Entry Points

- [x] `src/main.py`: replaced hardcoded `CONTEXT_LIMIT_CHARS` with `BUDGET = ContextBudget()`; soul and history calls use `BUDGET.soul_budget_chars()` / `BUDGET.history_budget_chars()`
- [x] `src/app.py`: same — budget-driven soul and history trimming per message; `was_trimmed` stored in `st.session_state.context_trimmed`
- [x] `OllamaClient._build_payload()` receives pre-assembled sections — deferred; current approach passes the combined string directly (sufficient until Phase 2 adds RAG)

### 1.9.7 — Streamlit Trim Indicator

- [x] After each message, if `st.session_state.get("context_trimmed")` is True, show a `⚠ Context trimmed` warning badge in the sidebar next to the message count caption
- [x] Clicking the badge opens the "📋 Logs" expander and switches to the "Context trim" tab — implemented via custom HTML + JS in `st.markdown(..., unsafe_allow_html=True)`; JS finds the `<details>` expander by summary text, opens it, then clicks the `[role=tab]` button matching "Context trim" after a 300 ms delay
- [x] Flag reflects only the most recent turn (recalculated on each message)

### 1.9.8 — Test Coverage (`tests/test_context.py`)

- [x] Test `count_tokens()` returns expected values for known strings
- [x] Test `ContextBudget.assemble()` with all sections populated — verify `was_trimmed` is False
- [x] Test with oversized misc section — verify it is truncated
- [x] Test vision trimmed before RAG when rag+vision budget is exceeded
- [x] Test response reserve is protected when `total_tokens == response_reserve`
- [x] Test `SoulFile.to_prompt_section()` drops `facts` first under tight budget
- [x] Test `to_prompt_section()` returns full text when no budget is set
- [x] All tests are pure-Python with no Ollama dependency — 10/10 passing

---

## Phase 1.95 — Memory Strategy & Intelligence

**Goal**: Define and implement the decision logic that governs *what* gets stored in short-term vs. long-term memory, rather than just *how* to store it. This phase codifies the memory policy so all later phases (Phase 2 RAG, soul file updates) follow consistent rules.

### 1.95.1 — Short-Term Memory Policy (within a session)

Short-term memory covers everything needed to maintain coherence in the current conversation — recent turns, clarifications, and established context.

- [x] Implement a **recency window** strategy in `trim_history()`: keep the last N messages or a token budget (Phase 1.9 budget drives the cap); drop or summarise older turns rather than blindly truncating — already implemented via `budget_chars` in Phase 1.9; `compress_history()` added to `client.py` for LLM-backed summarisation on overflow
- [x] Add a **relevance scoring** pre-filter before appending a turn to history: `is_filler_message()` in `src/memory/policy.py` detects filler turns (≤10 chars or exact filler phrase); wired into `app.py` — filler exchanges are shown in the UI but excluded from the history buffer
- [ ] Implement **entity/state tracking**: if a message updates a known entity (user name, preference, stated goal), replace the old value in the session state rather than appending a duplicate — deferred (requires NLP stack)
- [x] Implement **compression on overflow**: `compress_history()` in `src/llm/client.py` — when the recency window is full, summarises the oldest turn-pairs via `SUMMARISE_SESSION_PROMPT` into a single `[Earlier context summary]` system entry rather than discarding them; falls back to plain trim if Ollama is unavailable

### 1.95.2 — Long-Term Memory Policy (across sessions)

Long-term memory (soul file + Phase 2 RAG) should only persist durably useful information. The bar is deliberately higher than short-term.

- [x] Distinguish **explicit vs. implicit signals** in the soul patch prompt (`SOUL_PATCH_PROMPT`): prompt updated to require `_confidence` (0.0–1.0) and `_explicit` (bool) metadata keys in every patch; `_patch_worker` strips them before calling `apply_patch()`
- [x] Add a **novelty check** before writing to the soul file: already enforced by "NEVER STORE" rules and "no duplicates" instruction in `SOUL_PATCH_PROMPT`; `_is_repeat()` heuristic added to `should_store()` in `policy.py`
- [x] Apply a **stability heuristic**: `SOUL_PATCH_PROMPT` already instructs the model to store only things "still true in a year"; `_is_transient()` in `policy.py` adds a keyword-pattern check on candidate facts
- [x] Enforce a **confidence threshold**: `CONFIDENCE_THRESHOLD = 0.8` in `policy.py`; `_patch_worker` in `soul.py` skips low-confidence (< 0.8) non-explicit patches and logs the decision to `soul_changes.log`
- [ ] Implement **decay and review**: add a `last_reinforced` timestamp to each soul entry; flag entries not reinforced in 30 days — deferred (requires YAML schema changes and a review cron)
- [x] Define a **category whitelist** for long-term storage: `LONG_TERM_CATEGORIES` frozenset in `policy.py` covers `user_preferences`, `biographical_facts`, `relationships`, `domain_expertise`, `project_context`; enforced in `should_store()` gate 2

### 1.95.3 — Memory Decision Flow

Codify the go/no-go logic as a reusable utility so it can be called from both the soul-update path and the future RAG write path.

- [x] Implement `should_store(candidate: str, memory_store: dict, confidence: float) -> tuple[bool, str]` in `src/memory/policy.py`:
  1. **Is this new information, or a repeat?** → Return `(False, "repeat")` if it is already present
  2. **Is it about the user/task, or just incidental chat?** → Return `(False, "incidental")` if it does not fit any category in the whitelist
  3. **Is it likely to be useful in a future session?** → Return `(True, "short_term_only")` if it is transient
  4. **Is it stable and confident enough to trust?** → Return `(True, "long_term")` only if confidence exceeds the threshold; otherwise return `(True, "short_term_only")`
- [x] Write `tests/test_memory_policy.py` covering all four branches of the decision flow — 25/25 passing

### 1.95.4 — Two-Pass Extraction Architecture

- [x] Implement a **two-pass approach** for memory extraction: the main `OllamaClient` handles the conversation; a separate lightweight extraction call fires every `EXTRACT_EVERY = 5` turns (deliberately offset from `SOUL_UPDATE_EVERY = 3`) to decide what to commit to memory
- [x] Create `src/memory/extractor.py` with a `MemoryExtractor` class:
  - `extract_candidates(turn: dict) -> list[dict]` — sends `MEMORY_EXTRACT_PROMPT` to the LLM, returns a list of `{fact, category, confidence, explicit}` objects
  - `commit(candidates: list[dict], soul: SoulFile, memory_store)` — runs each candidate through `should_store()` and logs decisions to `data/logs/memory.log`
- [x] `MEMORY_EXTRACT_PROMPT` added to `src/llm/prompts.py`
- [x] Runs as daemon thread via `maybe_extract_memories()` — wired into `app.py`; never blocks the response path
- [x] Each extraction decision (candidate, decision, reason) logged to `data/logs/memory.log` via `get_logger("memory")`

---

## Phase 2 — Persistent Memory (RAG)

**Goal**: The LLM remembers information from previous conversations by retrieving relevant past interactions before each response.

### 2.1 — Set Up ChromaDB (`src/memory/vector_store.py`)

- [x] Add `chromadb` and `sentence-transformers` to `requirements.txt`
- [x] Create a `MemoryStore` class that wraps ChromaDB with persistence to `data/memory/`
- [x] Implement `add_memory(text: str, metadata: dict)` — embeds text and upserts it; ID is a 16-char SHA-256 hash of the content so re-adding the same summary is idempotent
- [x] Implement `query_memory(text: str, n_results: int = 5, threshold: float = 0.35) -> list[str]` — retrieves top-K memories whose cosine similarity meets the threshold; returns an empty list when nothing is relevant enough (so no `## Relevant Memory` block is injected)
- [x] Implement `clear_memory()` for development/testing
- [x] Write `tests/test_memory.py` — 14/14 passed (add, idempotency, count, empty/whitespace rejection, query relevance, threshold filtering, n_results cap, semantic ordering, clear, persistence across instances)

### 2.2 — Embeddings (`src/memory/embeddings.py`)

- [x] Download and cache `all-MiniLM-L6-v2` model from `sentence-transformers` — already cached locally from Phase 2.1 tests; no network needed
- [x] Create an `Embedder` class with an `embed(text: str) -> list[float]` method and `embed_batch(texts: list[str]) -> list[list[float]]`; model loaded lazily and cached with a module-level dict + lock so multiple instances share one load
- [x] Confirm embeddings are generated on CPU — `device="cpu"` hardcoded; matches the device used by `MemoryStore` so vectors are in the same embedding space
- [x] Write `tests/test_embeddings.py` — 15/15 passing (dimension=384, idempotency, empty rejection, semantic similarity ordering, batch, shared cache)

### 2.3 — Session Summariser (`src/memory/summariser.py`)

- [x] Write a `summarise_session(conversation_history, model, base_url, min_turns) -> str` function — returns `""` when fewer than `MIN_TURNS = 2` user turns (no LLM call, nothing to store); truncates transcripts over `MAX_CONV_CHARS = 12_000` from the front before sending
- [x] Feed the full conversation to the LLM with `SUMMARISE_SESSION_PROMPT`; strips system messages and `[Earlier context summary]` trim artefacts before forwarding
- [x] Return a concise paragraph capturing key facts, preferences, or information mentioned; returns `""` if the LLM response is whitespace-only so Phase 2.4 can skip the `add_memory()` call
- [x] Write `tests/test_summariser.py` — 19/19 passing (helper functions, short-circuit paths, mocked LLM path, truncation, and a live integration test skipped when Ollama is unavailable)

### 2.4 — Integrate Memory into Conversation Loop

- [x] On each user message: embed the message and query `MemoryStore` for top-5 relevant past session summaries
- [x] Only inject the `## Relevant Memory` block when at least one result clears the similarity threshold — skip entirely if nothing is relevant; Orion should not reach into long-term memory unless necessary
- [x] On conversation end (graceful exit): call `summarise_session()` and save the summary to `MemoryStore`
- [x] Wire `rag_budget_chars()` into `ContextBudget`; cap injected results by character budget (drop trailing results until total fits)
- [x] `app.py`: save session summary when "🗑️ Clear conversation" is clicked before resetting history
- [x] `main.py`: save session summary in `try/finally` so both graceful quit and Ctrl-C are covered
- [x] Add `## Relevant Memory` guideline to `BASE_SYSTEM_PROMPT` so the model knows how to use it
- [x] Log every `add_memory()` and `query_memory()` call to `data/logs/memory.log` (Phase 1.8.5 stub filled)
- [x] Write `tests/test_rag_integration.py` — 17 tests covering budget logic, injection, save-session, and memory logging

---

## Phase 3 — Voice Input (Speech-to-Text)

**Goal**: Replace the text input in the conversation loop with live microphone transcription.

### 3.1 — Install Dependencies

- [ ] Add `faster-whisper`, `sounddevice`, `numpy` to `requirements.txt`
- [ ] Confirm `faster-whisper` can access CUDA: run a short test transcription on GPU

### 3.2 — Whisper Transcription Module (`src/audio/stt.py`)

- [ ] Download and cache `faster-whisper` `small.en` model
- [ ] Create a `Transcriber` class that initialises the Whisper model on GPU (`device="cuda"`, `compute_type="float16"`)
- [ ] Implement `transcribe_file(path: str) -> str` for testing with audio files
- [ ] Implement `listen_and_transcribe() -> str`:
  - Capture audio from the default microphone using `sounddevice`
  - Enable built-in VAD (Voice Activity Detection) to detect end of speech
  - Return the transcribed string
- [ ] Add a `push_to_talk` mode (hold a key → record → release → transcribe) as an alternative to VAD
- [ ] Write `tests/test_audio.py` to test transcription accuracy with a sample WAV file

### 3.3 — Integrate STT into Main Loop

- [ ] Replace `input()` text prompt with `listen_and_transcribe()` call
- [ ] Print the transcribed text to the console so the user can verify it before the LLM responds
- [ ] Handle failed/empty transcriptions (silence, background noise) gracefully

---

## Phase 4 — Voice Output (Text-to-Speech)

**Goal**: The LLM's text responses are spoken aloud through a speaker.

### 4.1 — Install Piper TTS

- [ ] Install Piper TTS Python package: `pip install piper-tts` (or build from [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl))
- [ ] Download a Piper voice model (e.g., `en_US-amy-medium`) from the [Piper voices list](https://github.com/rhasspy/piper/blob/master/VOICES.md)
- [ ] Verify TTS works: synthesise "Hello, I am your robot" and play it through the speaker

### 4.2 — TTS Module (`src/audio/tts.py`)

- [ ] Create a `Speaker` class that wraps the Piper TTS engine
- [ ] Implement `speak(text: str)` — synthesise and play audio blocking until complete
- [ ] Implement streaming synthesis: split LLM response into sentences and synthesise each sentence as it arrives (reduces first-word latency significantly)
- [ ] Add a `mute` flag and a `set_voice(voice_model_path: str)` method
- [ ] Explore [Kokoro TTS](https://github.com/hexgrad/kokoro) as a higher-quality alternative and add it as a configurable backend

### 4.3 — Integrate TTS into Main Loop

- [ ] Replace `print(response)` with `speaker.speak(response)` after LLM responds
- [ ] Add a configurable option to enable/disable voice output (for silent/debug mode)
- [ ] Test the full voice loop: speak a question → STT transcribes → LLM responds → TTS speaks answer

---

## Phase 5 — Vision

**Goal**: The robot uses a camera to understand its environment and injects scene descriptions / object lists into the LLM's context window.

### 5.1 — Camera Capture (`src/vision/camera.py`)

- [ ] Add `opencv-python` to `requirements.txt`
- [ ] Create a `Camera` class that opens a USB webcam via `cv2.VideoCapture(0)`
- [ ] Implement `capture_frame() -> np.ndarray` — returns the latest BGR frame
- [ ] Implement `save_frame(path: str)` for debugging
- [ ] Add camera resolution configuration (default 640×480)

### 5.2 — Object Detection with YOLOv8 (`src/vision/detector.py`)

- [ ] Add `ultralytics` to `requirements.txt`
- [ ] Download `yolov8n.pt` (nano, ~6MB) or `yolov8s.pt` (small, ~22MB) for a speed/accuracy trade-off
- [ ] Create a `Detector` class with a `detect(frame: np.ndarray) -> list[str]` method
- [ ] Return a deduplicated, human-readable list of detected objects (e.g., `["person", "chair", "laptop"]`)
- [ ] Optionally draw bounding boxes on frames for debugging (`annotate_frame()`)
- [ ] Confirm CUDA is used for inference (check `device='cuda'` in YOLO init)

### 5.3 — Vision-Language Model (`src/vision/vlm.py`)

- [ ] Pull the Qwen2.5-VL model via Ollama: `ollama pull qwen2.5vl:7b`
- [ ] Create a `VisionLLM` class that sends an image + prompt to the Ollama vision API
- [ ] Implement `describe_scene(frame: np.ndarray) -> str` — returns a 1–2 sentence natural language description of the scene
- [ ] Implement `answer_visual_question(frame: np.ndarray, question: str) -> str`
- [ ] Test on a sample image; verify the description is coherent

### 5.4 — Integrate Vision into Main Loop

- [ ] Add a configurable `vision_interval` (e.g., capture a frame every 10 seconds)
- [ ] Build a `get_vision_context() -> str` function that:
  1. Captures a camera frame
  2. Runs YOLO detection for a fast object list
  3. Optionally calls the VLM for a richer scene description
  4. Returns a combined vision summary string
- [ ] Inject the vision context into the LLM system prompt under a `## Current Environment` section
- [ ] Test the full pipeline: point camera at objects → robot describes what it sees when asked

---

## Phase 5.5 — Face Recognition

**Goal**: Orion can detect faces in his camera feed and identify known people (Daniel and Danielle) by name, greet them on first sighting in a session, and ask who an unknown person is so he can learn them.

### 5.5.1 — Install Dependencies

- [ ] Add `face_recognition` and `dlib` to `requirements.txt`
- [ ] Add pre-built dlib wheel reference to `requirements_pi.txt` for Raspberry Pi aarch64 (avoids a slow source build)
- [ ] Verify `face_recognition` can locate faces on a test photo: `face_recognition.face_locations(image)`

### 5.5.2 — Face Enrollment

- [ ] Create `data/faces/` directory (gitignored — contains personal photos and encodings)
- [ ] Write a CLI enrollment script (`scripts/enroll_face.py`):
  - Takes `--name` and `--image` (or captures from webcam) as inputs
  - Computes the 128-d face encoding with `face_recognition.face_encodings()`
  - Saves `data/faces/<name>.npy` (numpy array of the encoding)
- [ ] Enroll Daniel and Danielle using good-quality reference photos
- [ ] Support multiple reference images per person (average their encodings for robustness)

### 5.5.3 — Face Recognition Module (`src/vision/faces.py`)

- [ ] Create a `FaceRecogniser` class that loads all `data/faces/*.npy` encodings at init time
- [ ] Implement `identify(frame: np.ndarray) -> list[str]` — detects all faces in the frame and returns a list of recognised names (or `"unknown"` for each unmatched face)
- [ ] Use a configurable Euclidean distance threshold (default `0.5`) — faces above the threshold are treated as unknown
- [ ] Implement `enroll(name: str, frame: np.ndarray)` — extracts the encoding from a live frame and saves it to `data/faces/<name>.npy`
- [ ] Add `recognition_interval` (e.g. every 5 seconds) to avoid running inference every frame

### 5.5.4 — LLM Integration

- [ ] On recognition, inject a `## People Present` section into the LLM context listing identified names
- [ ] Track a per-session `greeted` set so Orion greets each person only once per conversation (e.g. "Hey Daniel!" on first detection, then silent injection thereafter)
- [ ] Wire the greeting into TTS output (Phase 4) when available; fall back to print in CLI mode
- [ ] Update the system prompt guideline so Orion uses recognised names naturally in conversation

### 5.5.5 — Unknown Face Handling

- [ ] When `"unknown"` is returned and no greeting has been issued for that face region, have Orion ask: "I see someone I don't recognise — who are you?"
- [ ] If a name is provided in the response, call `FaceRecogniser.enroll(name, frame)` to save them
- [ ] Optionally update the soul file `user` or `facts` section with a note that a new person was met
- [ ] Implement a cooldown so Orion does not ask repeatedly for the same unknown face in one session

### 5.5.6 — Raspberry Pi Compatibility

- [ ] Verify `face_recognition` installs cleanly on Raspberry Pi OS 64-bit (Bookworm)
- [ ] Benchmark recognition latency on Pi 5 CPU; target <2 seconds per frame at `recognition_interval`
- [ ] If too slow, fall back to OpenCV Haar cascade for detection + `face_recognition` encoding only on detected regions (skips full-frame scan)

---

## Phase 5.6 — Network Bridge (Laptop Brain / Pi Body)

**Goal**: Keep all AI inference on the laptop while the Raspberry Pi handles only physical I/O (microphone, camera, speaker, motors). The Pi streams raw sensor data to the laptop over local Wi-Fi; the laptop does all the heavy lifting and streams results back. This unlocks the full pipeline with real hardware before committing to fully standalone Pi inference (Phase 6).

### 5.6.1 — Network Protocol Design

- [ ] Choose FastAPI + WebSocket for the laptop bridge server: WebSocket for bidirectional streaming (audio chunks ↔ transcription, LLM tokens ↔ text); HTTP POST for one-shot vision frames; HTTP GET for health checks
- [ ] Define the message schema for each channel:
  - **Audio** (`WebSocket /audio`): Pi sends raw 16 kHz / 16-bit mono PCM chunks; laptop returns `{"text": "..."}` on VAD-triggered end-of-utterance
  - **Vision** (`POST /vision`): Pi sends a JPEG body; laptop returns `{"summary": "..."}`
  - **Chat** (`WebSocket /chat`): Pi sends `{"text": "...", "vision_summary": "..."|null}`; laptop streams `{"delta": "..."}` tokens; ends with `{"done": true, "tool_call": {...}|null}`
  - **Health** (`GET /health`): returns `{"status": "ok", "model": "phi4-mini"}`
- [ ] Document the full protocol schema in `raspberry_pi/README.md`

### 5.6.2 — Laptop Bridge Server (`src/server.py`)

- [ ] Add `fastapi`, `uvicorn[standard]`, and `websockets` to `requirements.txt`
- [ ] Create `src/server.py` with a FastAPI app exposing all four endpoints above
- [ ] `GET /health` — returns server status and active Ollama model name
- [ ] `WebSocket /audio` — accumulates incoming PCM chunks; runs `Transcriber` with Silero VAD to detect end-of-speech; sends back the transcription event; handles silence and empty results gracefully
- [ ] `POST /vision` — accepts a JPEG body up to a configurable max size; calls `get_vision_context()` (Phase 5); returns the vision summary JSON
- [ ] `WebSocket /chat` — accepts the text + optional vision summary; runs the full laptop pipeline (`SoulFile`, `MemoryStore`, `ContextBudget`, `OllamaClient`); streams delta tokens; sends a final message including any motor `tool_call` extracted from the response
- [ ] Reuse existing modules (`OllamaClient`, `MemoryStore`, `SoulFile`, `ContextBudget`, `Transcriber`, vision pipeline) — no duplication of logic
- [ ] Add a `--host` and `--port` CLI flag (defaults: `0.0.0.0:8765`)
- [ ] Add a startup log message showing the server URL and active model
- [ ] Add request body size limits; reject oversized vision frames with HTTP 413

### 5.6.3 — Authentication & Configuration

- [ ] Add `python-dotenv` to `requirements.txt` and `requirements_pi.txt`
- [ ] Store a shared `ORION_AUTH_TOKEN` secret in `.env` (gitignored); load in `src/server.py` at startup
- [ ] Bridge server validates `Authorization: Bearer <token>` on every WebSocket handshake and HTTP request; rejects with 401 if missing or wrong
- [ ] Create `raspberry_pi/.env.example` with placeholder values for `ORION_SERVER_URL`, `ORION_AUTH_TOKEN`, and `VISION_INTERVAL`
- [ ] Pi bridge client reads token from its own `.env` (or `ORION_AUTH_TOKEN` environment variable)

### 5.6.4 — mDNS Service Discovery

- [ ] Add `zeroconf` to `requirements.txt` and `requirements_pi.txt`
- [ ] On server start, register a `_orion._tcp.local.` mDNS service record so the Pi can find the laptop by name without a hardcoded IP
- [ ] Write a `discover_server(timeout_s: float = 5.0) -> str | None` helper in `raspberry_pi/bridge_client.py` that queries mDNS for `_orion._tcp.local.` and returns the resolved `host:port`; returns `None` on timeout so the caller can fall back to `ORION_SERVER_URL`
- [ ] Test mDNS resolution from a second laptop before deploying to Pi

### 5.6.5 — Pi Bridge Client (`raspberry_pi/bridge_client.py`)

- [ ] Create `raspberry_pi/bridge_client.py` as the single entry point for hybrid mode; add a `--server` flag to override mDNS discovery with a hardcoded URL for debugging
- [ ] **Audio loop**: `sounddevice` captures 16 kHz PCM chunks → sends over `/audio` WebSocket → on `{"text": "..."}` received, opens `/chat` WebSocket with the transcribed text + latest cached vision summary
- [ ] **Chat loop**: receives streamed `delta` tokens → accumulates sentences → calls `Speaker.speak(sentence)` at each sentence boundary for low-latency playback; on `{"done": true}` message, extracts and executes any `tool_call` via `MotorController`
- [ ] **Vision loop**: background thread captures a frame every `VISION_INTERVAL` seconds (default 10 s) → HTTP POSTs to `/vision` → stores the returned summary in a thread-safe variable; latest cached summary is attached to every `/chat` request
- [ ] Graceful reconnect on WebSocket disconnect: exponential back-off starting at 1 s, capped at 30 s; log each attempt
- [ ] Skip motor execution silently if `gpiozero` is not installed (allows testing on non-Pi hardware)

### 5.6.6 — Audio Streaming Details

- [ ] Chunk size: 512 samples at 16 kHz = ~32 ms per chunk — keeps latency low while giving the server enough audio to run VAD
- [ ] Server accumulates chunks in a buffer until Silero VAD signals end-of-speech; transcribes the full buffer in one Whisper call
- [ ] Handle the case where VAD never fires (continuous background noise): flush and transcribe after a configurable `MAX_AUDIO_BUFFER_SECONDS = 30` timeout
- [ ] Return an empty `{"text": ""}` response (not an error) when transcription produces only whitespace; Pi discards empty transcriptions and does not open a `/chat` connection

### 5.6.7 — Motor Command Relay

- [ ] Server embeds a `tool_call` object in the final `/chat` message when the LLM response contains a movement command block:
  ```json
  {"done": true, "text": "Moving forward now!", "tool_call": {"name": "move", "direction": "forward", "duration_seconds": 2.0}}
  ```
- [ ] `tool_call` schema: `direction` must be one of `forward | backward | left | right | stop`; `duration_seconds` must be a positive float ≤ 10
- [ ] Server validates the schema before embedding; logs an error and omits `tool_call` on invalid LLM output rather than forwarding a malformed command
- [ ] Pi bridge client executes the command only after the full TTS sentence queue is cleared, so speech and movement do not start simultaneously

### 5.6.8 — Integration Tests & Simulation

- [ ] Write `tests/test_bridge.py` with the FastAPI `TestClient` and mocked `Transcriber`, `OllamaClient`, and vision pipeline; assert correct response shapes for all four endpoints
- [ ] Create `scripts/simulate_pi.py` — a laptop-based simulation script that connects to the local bridge server, sends a hardcoded transcript over `/chat`, and prints the streamed delta tokens to the console; useful for end-to-end smoke testing before Pi hardware is ready
- [ ] Add a `run_server` pytest fixture that starts the FastAPI app with `uvicorn` in a background thread for integration tests; tears down after each test module

### 5.6.9 — Documentation Update

- [ ] Update `raspberry_pi/README.md` with a "Hybrid Mode (Phase 5.6)" section covering:
  - How to start the bridge server on the laptop: `python src/server.py`
  - How to start the bridge client on the Pi: `python bridge_client.py`
  - Environment variable reference
  - Troubleshooting: mDNS not resolving, auth token mismatch, audio device selection

---

## Phase 6 — Physical Robot (Raspberry Pi)

**Goal**: Run a lightweight version of the entire pipeline on a Raspberry Pi 5 with physical motor control.

### 6.1 — Hardware Assembly

- [ ] Purchase Raspberry Pi 5 (8GB), robot chassis kit, L298N motor driver, and battery
- [ ] Wire L298N to Raspberry Pi GPIO and DC motors (see `raspberry_pi/README.md` for pinout diagram)
- [ ] Connect USB microphone, USB speaker, and camera to the Pi
- [ ] Verify 5V power is stable under load with motor test

### 6.2 — Raspberry Pi OS Setup

- [ ] Flash Raspberry Pi OS 64-bit (Bookworm) via Raspberry Pi Imager
- [ ] Enable SSH, camera interface, and I2C in `raspi-config`
- [ ] Update OS: `sudo apt update && sudo apt full-upgrade`
- [ ] Install Python 3.11 and create a virtual environment
- [ ] Write `raspberry_pi/setup.sh` to automate dependency installation

### 6.3 — llama.cpp on Raspberry Pi

- [ ] Clone [llama.cpp](https://github.com/ggerganov/llama.cpp) and compile with ARM NEON optimisations: `cmake -DLLAMA_NEON=ON ..`
- [ ] Download `Phi-3-mini-4k-instruct-q4.gguf` (Q4_K_M, ~2.2GB) from HuggingFace
- [ ] Test inference: `./main -m model.gguf -p "Hello" -n 100`
- [ ] Benchmark tokens/second; target >3 tok/s on the Pi 5
- [ ] Write a Python wrapper using `llama-cpp-python` to keep the interface consistent with the laptop client

### 6.4 — Lightweight STT and TTS on Pi

- [ ] Install `faster-whisper` on Pi; use `tiny.en` model (CPU inference)
- [ ] Install Piper TTS prebuilt binary for `aarch64` (Linux ARM64)
- [ ] Download a Piper voice model and verify audio output through the speaker
- [ ] Benchmark STT and TTS latency; target <3 seconds end-to-end for both

### 6.5 — Motor Control (`src/robot/motors.py`)

- [ ] Add `gpiozero` to `requirements_pi.txt`
- [ ] Create a `MotorController` class with methods: `forward(speed)`, `backward(speed)`, `turn_left(speed)`, `turn_right(speed)`, `stop()`
- [ ] Map L298N input pins to GPIO BCM pin numbers
- [ ] Write a test script to drive each motor independently and verify direction
- [ ] Add safety: automatically call `stop()` if the script exits or crashes

### 6.6 — LLM Tool Calling for Movement

- [ ] Define a simple movement tool schema:
  ```json
  {"name": "move", "parameters": {"direction": "forward|backward|left|right|stop", "duration_seconds": 1.0}}
  ```
- [ ] After each LLM response, check if the response contains a tool call JSON block
- [ ] Parse and execute motor commands from the LLM output
- [ ] Add natural language triggers: if the LLM says "move forward" in free text, also execute the command
- [ ] Write a test conversation that causes the robot to navigate a simple route

### 6.7 — Lightweight Vision on Pi

- [ ] Evaluate [Moondream2](https://huggingface.co/vikhyatk/moondream2) (1.8B VLM) for Pi vision — test inference speed
- [ ] If too slow, fall back to YOLOv8-nano (CPU) for object detection only
- [ ] Add option to offload vision inference to the laptop over local network (REST call)

### 6.8 — Integration and Auto-Start

- [ ] Write `raspberry_pi/main_pi.py` as a single entry point combining STT → LLM → TTS → vision → motors
- [ ] Create a `systemd` service file so the robot starts automatically on boot
- [ ] Test full pipeline: turn on Pi → robot boots → greets user → takes voice commands → moves around
- [ ] Add a hardware kill switch (physical button on GPIO to safely shut down)

---

## Stretch Goals (Post Phase 6)

- [ ] **Obstacle avoidance**: Add an HC-SR04 ultrasonic distance sensor and autonomous stop-on-obstacle logic
- [ ] **Wake word detection**: Replace push-to-talk with an always-on wake word (e.g., "Hey Robot") using [openWakeWord](https://github.com/dscripka/openWakeWord)
- [ ] **Navigation mapping**: Use a simple grid map to track the robot's known environment
- [ ] **Fine-tuning**: Collect conversation logs and fine-tune Phi-4-mini on personal vocabulary/preferences using LoRA via `peft` + `trl`
- [ ] **Multi-robot**: Run a second Pi robot and have the two communicate over local MQTT
- [ ] **Web UI**: Build a simple local web dashboard (FastAPI + HTML) to view logs, memory, and camera feed
- [ ] **Battery monitor**: Read battery voltage via ADC and have the robot announce when it needs charging
