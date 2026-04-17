# Memory System

Orion's memory is split across two complementary tiers: a **structured soul file** that holds identity and biographical facts, and a **semantic vector store** that holds episodic summaries of past conversations. They solve different problems and are written through different pipelines.

---

## Architecture overview

```
                     Per-message path (sync, on every turn)
                 ┌────────────────────────────────────────────┐
  User message ──► is_filler_message()  ──► history buffer
                 │                              │
                 │  RAG query (non-filler only) ▼
                 │  MemoryStore.query_memory() ──► ## Relevant Memory
                 │                              │     injected into
                 │  SoulFile.to_prompt_section() ──► system prompt
                 └────────────────────────────────────────────┘

                     Background daemon threads (fire-and-forget)
                 ┌────────────────────────────────────────────┐
                 │  every 3 turns → maybe_update_soul()       │
                 │  every 5 turns → maybe_extract_memories()  │
                 │  every 6 turns → maybe_grow_curiosity()    │
                 │  on clear/exit → summarise_session()       │
                 │                  MemoryStore.add_memory()  │
                 └────────────────────────────────────────────┘
```

---

## Files

| File | Role |
|---|---|
| `soul.py` | Read/write wrapper for `data/soul.yaml`; budget-aware trimming; soul patch loop |
| `vector_store.py` | ChromaDB-backed persistent memory; `add_memory` / `query_memory` |
| `summariser.py` | Converts a completed conversation into a paragraph for long-term storage |
| `extractor.py` | Daemon thread that extracts factual candidates from each user turn |
| `policy.py` | Filler detection and four-gate should-store decision logic |
| `embeddings.py` | Standalone `Embedder` wrapper for sentence-transformers vectors |

---

## Tier 1 — Soul file (`soul.py`)

### What it stores

`data/soul.yaml` is a structured YAML document with top-level sections. The sections in the live file are:

| Section | Contents |
|---|---|
| `identity` | Name, persona, communication style, capabilities, personality notes, curiosity queue |
| `user` | Name, DOB, location, profession, education, skills, interests |
| `partner` | Name, relationship, occupation, interests |
| `environment` | Physical location, hardware |
| `facts` | Learned facts that don't fit another section |

### How it is updated

Every **3 user turns**, `maybe_update_soul()` spawns a background daemon thread. The thread re-runs the last portion of the conversation through the LLM using a structured extraction prompt. If the LLM responds with a `json` fenced block it is parsed as a patch dict shaped `{section: {key: value}}` and merged into the file via `apply_patch()`.

`apply_patch()` performs a **deep merge**: if both the existing value and the incoming value are dicts (e.g. `identity.personality_notes`), keys are merged rather than overwritten. This lets successive patches accumulate detail without clobbering earlier entries.

Separately, every **6 user turns**, `maybe_grow_curiosity()` asks the LLM to generate new questions to add to `identity.curiosity_queue`.

### Atomic writes

All writes go through a module-level `threading.Lock`. The actual write pattern is:

1. Dump YAML to a `.yaml.tmp` sibling file.
2. `tmp.replace(soul_path)` — atomic on all supported platforms.

This guarantees no reader ever sees a partial write.

### System prompt injection

`SoulFile.to_prompt_section(budget_chars=N)` renders the YAML as a `## About Me` markdown code block and injects it into the system prompt. When the rendered text exceeds the character budget, sections are progressively dropped — least important first:

| Level | Action |
|---|---|
| 1 | Drop `facts` |
| 2 | Drop `environment` |
| 3 | Trim `user` to `{name, preferred_name, date_of_birth, location}` |
| 4 | Trim `identity` to `{name, persona, communication_style}` |
| 5 | Trim `partner` to `{name, preferred_name, relationship}` |

Each drop is tried in sequence and the function returns as soon as the text fits. All trim events are logged to `data/logs/soul_changes.log`.

---

## Tier 2 — Vector memory store (`vector_store.py`)

### Technology

- **ChromaDB** (`PersistentClient`) stores documents in `data/memory/`.
- **`sentence-transformers` `all-MiniLM-L6-v2`** (22 MB, CPU-only) provides embeddings.  The GPU is reserved for Ollama LLM inference.
- The ChromaDB collection uses **cosine similarity** (`hnsw:space: cosine`).

### Document IDs

Document IDs are the first 16 hex characters of the **SHA-256 hash of the text**. This makes every `add_memory()` call idempotent: storing the same summary twice performs an upsert instead of creating a duplicate.

### Query threshold

`query_memory(text, n_results=5, threshold=0.35)` retrieves up to `n_results` candidate documents from ChromaDB and then filters out anything whose cosine similarity is below `threshold`. The default 0.35 is intentionally permissive — it errs toward recalling something marginally relevant rather than silently returning nothing. If nothing clears the threshold the method returns `[]` and no `## Relevant Memory` block is injected.

ChromaDB returns **distance** (`1 − similarity`), so the conversion is:

```
similarity = 1.0 - distance
keep if similarity >= threshold
```

### Thread safety

A `threading.Lock` guards all `add_memory` and `query_memory` calls so the background summariser thread and the foreground query path never race on the ChromaDB client.

---

## How memories enter the store — the write pipeline

There are two write paths into the vector store:

### Path 1 — Session summary (primary)

When a session ends (`Clear conversation` button in the UI, or `quit` in the CLI), `summarise_session()` feeds the full conversation transcript to the LLM using `SUMMARISE_SESSION_PROMPT` and receives a concise 3–5 sentence paragraph. The paragraph is then persisted via `MemoryStore.add_memory(summary, {"source": "session_summary"})`.

Constraints:
- Requires at least **2 user turns** (`MIN_TURNS = 2`); single-exchange sessions are skipped.
- Conversation text is capped at **12,000 characters** (oldest turns truncated first) so the summarisation prompt fits within the model's context window.
- The summary is stored with `source: session_summary` metadata for auditability.

### Path 2 — Turn-level extraction (background)

Every **5 user turns**, `maybe_extract_memories()` spawns a daemon thread. The thread calls the LLM with `MEMORY_EXTRACT_PROMPT` on the last user message and expects a JSON block like:

```json
{
  "candidates": [
    {"fact": "Daniel prefers dark roast coffee.", "category": "user_preferences", "confidence": 0.92, "explicit": true}
  ]
}
```

Each candidate then passes through the **four-gate policy** in `policy.py`:

| Gate | Check | Outcome on failure |
|---|---|---|
| 1 — Repeat | Substring search against soul dict | `"repeat"` — skip |
| 2 — Category | Keyword match against `LONG_TERM_CATEGORIES` signals | `"incidental"` — skip |
| 3 — Transience | Regex for time-limited words (`today`, `right now`, `weather` …) | `"short_term_only"` — log only |
| 4 — Confidence | `confidence >= 0.8` | `"short_term_only"` if below |

Currently the extractor **logs** its decisions to `data/logs/memory.log` but does not yet commit candidates directly to the vector store (wiring in a future phase). Session summaries are the active write path.

---

## How memories surface in context — the read pipeline

On every non-filler user message, immediately before constructing the LLM payload:

```python
rag_results = memory_store.query_memory(user_input)
```

Results are assembled into a `## Relevant Memory` section and appended to the system prompt. A budget cap is enforced:

```python
rag_budget = BUDGET.rag_budget_chars()   # 6144 chars (20% of 7680 usable tokens × 4)
for result in rag_results:
    entry = f"- {result}"
    if total_chars + len(entry) + 1 > rag_budget:
        break
    kept.append(entry)
```

If `rag_results` is empty the section is omitted entirely, rather than injecting an empty heading.

**Filler suppression**: `is_filler_message(user_input)` is checked *before* the query. Short acknowledgements ("ok", "thanks", "got it") and messages under 10 characters never trigger a vector lookup, which saves embedding time and avoids polluting the prompt with irrelevant results triggered by a one-word response.

The final assembled system prompt has this shape:

```
{BASE_SYSTEM_PROMPT}

## About Me
```yaml
<soul YAML>
```

## Current Time
...

## Relevant Memory
- <session summary sentence>
- <session summary sentence>
```

---

## Context budget

Memory injection is governed by `ContextBudget` in `src/llm/context.py`. The default allocation for an 8192-token model:

| Tier | Share | Characters |
|---|---|---|
| Soul / system | 20 % | 6 144 |
| History | 50 % | 15 360 |
| RAG + vision | 20 % | 6 144 |
| Misc (time etc.) | 10 % | 3 072 |
| Response reserve | — | 512 tokens |

The soul trimmer and history trimmer each receive their own per-tier budget. The RAG budget cap loop enforces the RAG tier maximum inline.

---

## Logging

| Log file | Written by | Contains |
|---|---|---|
| `data/logs/memory.log` | `vector_store.py`, `extractor.py` | `add_memory` / `query_memory` calls; extractor candidate decisions |
| `data/logs/soul_changes.log` | `soul.py` | Every soul patch (before/after per key); trim level events |

All loggers are created lazily on first use via `src/utils/log.get_logger()`. They use `propagate=False` so they write exclusively to their own rotating file handlers and never pollute stdout or pytest's caplog.

---

## Data directory layout

```
data/
  soul.yaml          ← structured identity/facts YAML (human-editable)
  memory/            ← ChromaDB persistent store (binary, not hand-edited)
    chroma.sqlite3
    ...
  logs/
    memory.log       ← vector store operations
    soul_changes.log ← soul patch audit trail
    context_trim.log ← triggered when history or soul sections are truncated
    conversations/
      <date>.jsonl   ← per-session conversation transcripts
```

---

## Adding new facts manually

The soul file can be edited directly in `data/soul.yaml`. The UI provides a live YAML viewer in the sidebar ("Soul file" expander) and a "Force soul update now" button that triggers an immediate background patch check against the current conversation.

To add a memory directly to the vector store from a Python REPL or script:

```python
from src.memory.vector_store import MemoryStore

store = MemoryStore()
store.add_memory(
    "Daniel is learning Rust in his spare time.",
    {"source": "manual", "date": "2026-04-16"},
)
```

To inspect what is in the store:

```python
results = store.query_memory("What is Daniel learning?")
print(results)
```
