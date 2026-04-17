# Memory System

Orion's memory is split across three complementary tiers: a **lean soul file** that holds always-on identity and core biographical facts, a **structured facts store** that holds detailed knowledge retrieved on demand, and a **semantic vector store** that holds episodic summaries of past conversations.

---

## Architecture overview

```
                     Per-message path (sync, on every turn)
                 ┌──────────────────────────────────────────────────────┐
  User message ──► is_filler_message()  ──► history buffer
                 │                              │
                 │  RAG query (non-filler only) ▼
                 │  MemoryStore.query_memory() ──► ## Relevant Memory   │
                 │  FactsStore.query_facts()   ──► ## Relevant Facts    │
                 │  SoulFile.to_prompt_section() ──► ## About Me        │
                 │                              all injected into        │
                 │                              system prompt            │
                 └──────────────────────────────────────────────────────┘

                     Background daemon threads (fire-and-forget)
                 ┌──────────────────────────────────────────────────────┐
                 │  every 3 turns → maybe_update_soul()                 │
                 │  every 5 turns → maybe_extract_memories()            │
                 │                  → structured facts → FactsStore     │
                 │                  → episodic facts   → MemoryStore    │
                 │  every 6 turns → maybe_grow_curiosity()              │
                 │  on clear/exit → summarise_session()                 │
                 │                  MemoryStore.add_memory()            │
                 └──────────────────────────────────────────────────────┘

                     On startup (once per session)
                 ┌──────────────────────────────────────────────────────┐
                 │  migrate_soul_to_facts() — trims soul to core fields  │
                 │  and moves non-core data into FactsStore (idempotent) │
                 └──────────────────────────────────────────────────────┘
```

---

## Files

| File | Role |
|---|---|
| `soul.py` | Read/write wrapper for `data/soul.yaml`; budget-aware trimming; soul patch loop; migration helper |
| `facts_store.py` | JSON-backed structured facts store; category + keyword retrieval |
| `vector_store.py` | ChromaDB-backed persistent memory; `add_memory` / `query_memory` |
| `summariser.py` | Converts a completed conversation into a paragraph for long-term storage |
| `extractor.py` | Daemon thread that extracts factual candidates and routes them to the appropriate store |
| `policy.py` | Filler detection and four-gate should-store decision logic |
| `embeddings.py` | Standalone `Embedder` wrapper for sentence-transformers vectors |

---

## Tier 1 — Soul file (`soul.py`)

### What it stores

`data/soul.yaml` is intentionally lean. It holds only the information Orion genuinely needs in every single message — never retrieved, always present.

| Section | Contents |
|---|---|
| `identity` | Name, persona, communication style, capabilities, hardware, personality notes, curiosity queue |
| `user` | Name, preferred name, date of birth, location |
| `partner` | Name, preferred name, relationship |

Everything else (profession, education, interests, travel, partner's career, etc.) lives in the facts store.

### `identity` — always grows

The `identity` section is never trimmed or migrated. Two sub-keys expand automatically over time:

- `identity.personality_notes` — observations, opinions, and character traits Orion develops through conversation. The soul patch LLM writes here whenever something resonates.
- `identity.curiosity_queue` — questions Orion genuinely wants to ask Daniel. Grown every 6 turns by `maybe_grow_curiosity()`. Questions are consumed by the curiosity threshold system in `app.py` and asked one at a time when the moment feels right.

### How it is updated

Every **3 user turns**, `maybe_update_soul()` spawns a background daemon thread. The `SOUL_PATCH_PROMPT` is scoped to only write to:
- `user` — name, preferred_name, date_of_birth, location
- `user.family` — immediate relatives (parents, siblings)
- `partner` — name, preferred_name, relationship
- `identity.personality_notes`

Any other facts discovered in conversation are extracted separately by `maybe_extract_memories()` and written to the facts store.

### Atomic writes

All writes go through a module-level `threading.Lock`. The write pattern is:

1. Dump YAML to a `.yaml.tmp` sibling file.
2. `tmp.replace(soul_path)` — atomic on all supported platforms.

### System prompt injection

`SoulFile.to_prompt_section(budget_chars=N)` renders the YAML as a `## About Me` markdown code block. Because the soul is now lean this rarely needs trimming, but the progressive drop logic is retained as a safety net:

| Level | Action |
|---|---|
| 1 | Drop `facts` (legacy — should be empty after migration) |
| 2 | Drop `environment` |
| 3 | Trim `user` to `{name, preferred_name, date_of_birth, location}` |
| 4 | Trim `identity` to `{name, persona, communication_style}` |
| 5 | Trim `partner` to `{name, preferred_name, relationship}` |

---

## Tier 2 — Structured facts store (`facts_store.py`)

### What it stores

`data/facts.json` holds durable, structured facts that are too detailed for the always-on soul file but too well-defined for the unstructured vector store. Facts are grouped into categories.

| Category | Contents |
|---|---|
| `work` | Job title, employer, department, focus areas, tools, notable projects |
| `education` | Degrees, institutions, dissertation topics |
| `interests` | Hobbies, sports, music, games, entertainment |
| `skills` | Programming languages, platforms, frameworks |
| `relationships` | Extended family, named friends and connections |
| `partner` | Partner's career, education, studio, exhibitions, awards |
| `travel` | Places lived, countries visited |
| `projects` | Personal and open-source projects |
| `general` | Anything durable that does not fit another category |

### File layout

```json
{
  "work": [
    {
      "fact": "Daniel is a Data Scientist Manager at TJX Companies.",
      "confidence": 1.0,
      "explicit": true,
      "source": "migration",
      "ts": "2026-04-17T10:00:00"
    }
  ],
  ...
}
```

### Retrieval — two-pass

`FactsStore.query_facts(message)` runs two passes:

1. **Category pass** — a trigger-keyword map routes the message to one or more categories. All facts in matched categories are returned.
2. **Keyword fallback** — scans facts in remaining categories for any word (≥4 chars) from the message that appears in the fact text.

Results are deduplicated, capped at `max_facts=20`, and injected as a `## Relevant Facts` block in the system prompt (budget-capped at half the RAG tier allocation).

### Writes

Facts are written by two paths:
- **Migration** — `migrate_soul_to_facts()` on startup (idempotent).
- **Extractor** — `maybe_extract_memories()` background thread routes structured categories here (see Tier 3 write pipeline below).

`add_fact()` and `add_facts_bulk()` both deduplicate via case-insensitive substring matching, so re-adding the same fact is always a no-op.

---

## Tier 3 — Vector memory store (`vector_store.py`)

### Technology

- **ChromaDB** (`PersistentClient`) stores documents in `data/memory/`.
- **`sentence-transformers` `all-MiniLM-L6-v2`** (22 MB, CPU-only) provides embeddings. The GPU is reserved for Ollama LLM inference.
- The ChromaDB collection uses **cosine similarity** (`hnsw:space: cosine`).

### Document IDs

Document IDs are the first 16 hex characters of the **SHA-256 hash of the text**. This makes every `add_memory()` call idempotent: storing the same summary twice performs an upsert instead of creating a duplicate.

### Query threshold

`query_memory(text, n_results=5, threshold=0.35)` retrieves up to `n_results` candidate documents and filters out anything whose cosine similarity is below `threshold`. If nothing clears the threshold, `[]` is returned and no `## Relevant Memory` block is injected.

```
similarity = 1.0 - distance   (ChromaDB returns distance, not similarity)
keep if similarity >= 0.35
```

### Thread safety

A `threading.Lock` guards all `add_memory` and `query_memory` calls.

---

## How memories enter the stores — the write pipeline

### Path 1 — Session summary → vector store (primary)

When a session ends (`Clear conversation` button), `summarise_session()` produces a concise 3–5 sentence paragraph from the full conversation transcript. The paragraph is stored via `MemoryStore.add_memory(summary, {"source": "session_summary"})`.

Constraints:
- Requires at least **2 user turns** (`MIN_TURNS = 2`).
- Conversation text is capped at **12,000 characters** (oldest turns truncated first).

### Path 2 — Turn-level extraction → facts store or vector store

Every **5 user turns**, `maybe_extract_memories()` spawns a daemon thread. The thread calls the LLM with `MEMORY_EXTRACT_PROMPT` and receives candidates:

```json
{
  "candidates": [
    {"fact": "Daniel prefers dark roast coffee.", "category": "user_preferences", "confidence": 0.92, "explicit": true}
  ]
}
```

Each candidate passes through the **four-gate policy** in `policy.py`:

| Gate | Check | Outcome on failure |
|---|---|---|
| 1 — Repeat | Substring search against soul dict | `"repeat"` — skip |
| 2 — Category | Keyword match against `LONG_TERM_CATEGORIES` | `"incidental"` — skip |
| 3 — Transience | Regex for time-limited words (`today`, `right now` …) | `"short_term_only"` — log only |
| 4 — Confidence | `confidence >= 0.8` | `"short_term_only"` if below |

Facts that pass all four gates are then **routed by category**:

| Route | Categories | Destination |
|---|---|---|
| Structured | `work`, `education`, `interests`, `skills`, `relationships`, `partner`, `travel`, `projects`, `general`, `user_preferences`, `biographical_facts`, `domain_expertise`, `project_context` | `FactsStore` |
| Episodic | anything else | `MemoryStore` (vector store) |

---

## How memories surface in context — the read pipeline

On every non-filler user message, immediately before constructing the LLM payload, three injections happen:

```python
# 1. Episodic memory
rag_results = memory_store.query_memory(user_input)
# → ## Relevant Memory block

# 2. Structured facts
facts_results = facts_store.query_facts(user_input)
# → ## Relevant Facts block

# 3. Soul (always present)
soul_section = soul.to_prompt_section(budget_chars=BUDGET.soul_budget_chars())
# → ## About Me block
```

**Filler suppression**: RAG and facts queries are skipped for filler messages (`is_filler_message()` returns True) — short acknowledgements never trigger lookups.

The assembled system prompt has this shape:

```
{BASE_SYSTEM_PROMPT}

## About Me
```yaml
<lean soul YAML>
```

## Current Time
...

## Relevant Memory
- <episodic summary sentence>

## Relevant Facts
- <structured fact>
- <structured fact>
```

---

## Context budget

Memory injection is governed by `ContextBudget` in `src/llm/context.py`.

| Tier | Share | Characters |
|---|---|---|
| Soul / system | 35 % | ~10 752 |
| History | 35 % | ~10 752 |
| RAG + vision | 20 % | ~6 144 |
| Misc (time etc.) | 10 % | ~3 072 |
| Response reserve | — | 512 tokens |

The `## Relevant Facts` block shares the RAG+vision budget, capped at half of it (≈3 072 chars) so both sections can coexist.

---

## Soul → facts migration

`migrate_soul_to_facts(soul)` in `soul.py` is called once per session on app startup. It:

1. Reads all non-core fields from `user` and `partner` sections.
2. Flattens nested YAML structures into plain-English fact strings via `_flatten_to_facts()`.
3. Bulk-inserts them into the `FactsStore` (deduplication means repeated calls are safe).
4. Trims `user` to `{name, preferred_name, date_of_birth, location}` and `partner` to `{name, preferred_name, relationship}`.
5. Removes the legacy `facts` section entirely.
6. Saves the trimmed soul file atomically.

---

## Logging

| Log file | Written by | Contains |
|---|---|---|
| `data/logs/memory.log` | `vector_store.py`, `extractor.py`, `facts_store.py` | `add_memory` / `query_memory` / `add_fact` calls; extractor routing decisions |
| `data/logs/soul_changes.log` | `soul.py` | Every soul patch (before/after per key); trim level events; migration summary |
| `data/logs/context_trim.log` | `context.py` | Triggered when history or soul sections are truncated |

All loggers are created lazily on first use via `src/utils/log.get_logger()` and write exclusively to their own rotating file handlers.

---

## Data directory layout

```
data/
  soul.yaml          ← lean identity YAML: identity, user (core), partner (core)
  facts.json         ← structured facts store: work, education, interests, etc.
  memory/            ← ChromaDB persistent store (binary, not hand-edited)
    chroma.sqlite3
    ...
  logs/
    memory.log       ← vector store + extractor + facts store operations
    soul_changes.log ← soul patch audit trail + migration log
    context_trim.log ← triggered when history or soul sections are truncated
    conversations/
      <date>.jsonl   ← per-session conversation transcripts
```

---

## Adding new data manually

**Soul file** — edit `data/soul.yaml` directly. The UI provides a live YAML viewer in the sidebar ("Soul file" expander) and a "Force soul update now" button.

**Facts store** — edit `data/facts.json` directly, or from a Python REPL:

```python
from src.memory.facts_store import FactsStore

store = FactsStore()
store.add_fact(
    "Daniel is learning Rust in his spare time.",
    category="interests",
    source="manual",
)

# Query what's known about a topic
results = store.query_facts("What is Daniel learning?")
print(results)
```

**Vector store** — from a Python REPL:

```python
from src.memory.vector_store import MemoryStore

store = MemoryStore()
store.add_memory(
    "Daniel mentioned he wants to build a chess engine as a side project.",
    {"source": "manual", "date": "2026-04-17"},
)

results = store.query_memory("What projects is Daniel thinking about?")
print(results)
```

