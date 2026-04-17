# Personal Robot

A fully local, open-source robot system — built in phases from a conversational AI running on a laptop all the way to a self-contained physical robot on a Raspberry Pi.

No cloud APIs. No subscriptions. Everything runs on your hardware.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Phase Summary](#phase-summary)
- [Hardware Requirements](#hardware-requirements)
- [Software Stack](#software-stack)
- [Models](#models)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Raspberry Pi Deployment](#raspberry-pi-deployment)
- [Contributing](#contributing)

---

## Project Overview

This project builds a personal robot from the ground up using only open-source tools and local inference. It is structured in six progressive phases:

1. **Local LLM** — Run a small language model on a laptop GPU (NVIDIA RTX 4060)
2. **Persistent Memory** — Give the LLM long-term memory using a local RAG (Retrieval-Augmented Generation) system
3. **Voice Input** — Transcribe spoken words from a microphone in real time
4. **Voice Output** — Have the robot speak responses through a speaker
5. **Vision** — Identify surroundings using a camera and feed context to the LLM
5.5. **Face Recognition** — Detect and identify known people (Daniel, Danielle) and greet them by name
6. **Physical Robot** — Deploy everything to a Raspberry Pi controlling motors and wheels

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Laptop / Dev Machine             │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │ Microphone│───▶│  STT     │───▶│              │  │
│  │ (USB)    │    │ (Whisper)│    │   LLM Core   │  │
│  └──────────┘    └──────────┘    │  (Phi-4-mini)│  │
│                                  │              │  │
│  ┌──────────┐    ┌──────────┐    │              │  │
│  │  Camera  │───▶│  Vision  │───▶│              │  │
│  │ (USB)    │    │ (YOLO /  │    └──────┬───────┘  │
│  └──────────┘    │  VLM)    │           │          │
│                  └──────────┘    ┌──────▼───────┐  │
│                                  │   Memory     │  │
│  ┌──────────┐    ┌──────────┐    │  (ChromaDB   │  │
│  │ Speaker  │◀───│  TTS     │    │   + RAG)     │  │
│  │ (USB)    │    │ (Piper)  │    └──────────────┘  │
│  └──────────┘    └──────────┘                      │
└─────────────────────────────────────────────────────┘
         │
         │  Phase 6: Deploy to Raspberry Pi
         ▼
┌─────────────────────────────────────────────────────┐
│                  Raspberry Pi 5                      │
│  Mic → Whisper (tiny) → LLM (llama.cpp) → Piper TTS │
│  Camera → Vision → LLM context                      │
│  GPIO → Motor Driver (L298N) → Wheels               │
└─────────────────────────────────────────────────────┘
```

---

## Phase Summary

### ✅ Phase 0 — Environment Setup *(complete)*

- Python 3.11.9, `.venv`, Ollama v0.20.7, FFmpeg v8.1 installed
- NVIDIA RTX 4060 Laptop GPU confirmed (driver 596.21, 8GB VRAM)
- Project folder structure, `.gitignore`, and `requirements.txt` in place

### ✅ Phase 1 — Local LLM *(complete)*

Run a top-ranked Hugging Face small language model entirely on the local GPU with no internet required.

- **Model**: [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/phi-4-mini-instruct) (3.8B parameters, Q4_K_M, 2.5GB) — GPU inference confirmed (+3,403 MiB VRAM)
- **Alternative pulled**: `llama3.2:3b` (Meta, 2.0GB) available for comparison
- **Framework**: [Ollama](https://ollama.com/) v0.20.7 with [llama.cpp](https://github.com/ggerganov/llama.cpp) CUDA backend
- **Client**: `OllamaClient` wrapper in `src/llm/client.py` with streaming, history trimming, and configurable system prompt
- **CLI loop**: `src/main.py` — multi-turn conversation with context window management
- **Tests**: `tests/test_llm.py` — 7/7 passing

### ✅ Phase 1.5 — Streamlit Chat UI *(complete)*

A local web app for testing multi-turn conversations without the CLI.

- **Entry point**: `src/app.py` — run with `streamlit run src/app.py`
- **Features**: streaming chat bubbles, persistent session history, sidebar model selector (live from Ollama), editable system prompt, clear conversation button, online/offline status badge
- **Performance**: Ollama availability and model list are cached (`ttl=10s` / `ttl=30s`) to avoid redundant HTTP calls on re-renders
- **Verified**: model switching mid-session (`phi4-mini` ↔ `llama3.2:3b`) works without error

### ✅ Phase 1.7 — Soul File (Identity & Long-Term Facts) *(complete)*

Give the LLM a persistent, human-readable YAML file that defines its identity and stores facts it learns over time.

- **File**: `data/soul.yaml` — sections for `identity`, `user`, `environment`, and `facts`; gitignored so personal data stays local
- **Module**: `src/memory/soul.py` — `SoulFile` class with `load()`, `to_prompt_section()`, `update()`, and `apply_patch()`
- **Self-updating**: after every N turns, the LLM is asked whether it learned anything worth keeping; if it outputs a JSON patch block, it is automatically written back to the soul file
- **Injection**: the soul file is loaded on startup and injected into the system prompt as a `## About Me` section, so the LLM always has its identity and known facts in context
- **Scope — identity layer only**: the soul file is the *always-on* layer injected into every prompt. It should stay compact and contain only stable, perpetually-relevant facts (Orion's identity, Daniel's core profile). Episodic knowledge — what was discussed in past sessions, situational preferences, one-off facts — belongs in Phase 2 RAG, not in the soul file's `facts` section.
- **Distinct from RAG**: the soul file is structured and hand-editable; Phase 2 RAG handles unstructured episodic search over conversation history

### ✅ Phase 1.8 — Logging & Observability *(complete)*

A unified logging layer that writes structured, human-readable records for every significant event — so changes to the soul file, the in-context conversation window, and the long-term memory store can all be reviewed after the fact.

- **Central log setup** (`src/utils/log.py`): single `get_logger(name)` factory that writes to both a rotating file under `data/logs/` and the console (DEBUG to file, WARNING to console). All modules import from here instead of configuring their own handlers.
- **Conversation log** (`data/logs/conversations/`): every turn is appended to a per-session `.jsonl` file — one JSON object per line containing timestamp, role, content, and active model. Lets you replay any past session exactly as it happened.
- **Soul file audit log** (`data/logs/soul_changes.log`): whenever a patch is applied, the full before/after diff is recorded. Replaces the existing unstructured `soul_worker.log` with a richer, diff-based format.
- **Short-term memory log** (`data/logs/context_trim.log`): each time the rolling conversation history is trimmed, log the messages that were dropped and the new history length. Makes context window behaviour visible.
- **Long-term memory log** (`data/logs/memory.log`): logs every ChromaDB query (top-K results + similarity scores) and every session summary write. Entries record whether the relevance threshold was cleared and what was injected.
- **Streamlit log viewer**: the existing "Worker log" sidebar expander is replaced with a tabbed viewer covering all four log files, with per-log clear buttons.
- **Test coverage**: `tests/test_logging.py` — 12/12 passing; verifies conversation, soul, trim, and memory log files are created and populated correctly using `tmp_path` fixtures; no live LLM required.

### ✅ Phase 1.9 — Context Budget Management *(complete)*

A budget-aware context assembly layer that measures every section before sending it to the LLM, ensures the model always has headroom to reply, and gracefully degrades the least-important sections rather than silently overflowing.

#### Strategy

The effective context window is divided into percentage-based tiers, calculated at assembly time on every message:

| Tier | Allocation | Contents |
|---|---|---|
| Response reserve | 512 tokens (hard floor) | Always protected — guaranteed headroom for the LLM's reply |
| System + soul | 20% of remaining | `BASE_SYSTEM_PROMPT` + soul file (`## About Me`) |
| Conversation history | 50% of remaining | Rolling message list, trimmed from oldest first |
| RAG + vision | 20% of remaining | `## Relevant Memory` + `## Current Environment` |
| Time + misc | 10% of remaining | `## Current Time` and any future injected sections |

> Rough numbers at the working 4 096-token cap: ~660 tokens for soul/system, ~1 640 for history, ~660 for RAG+vision, ~330 for time. Reserve is always 512 on top.

#### Priority and Degradation

When a tier is over-budget, sections are cut in this order (highest priority survives longest):

1. **Soul file** — trim `facts` and `environment` sections first; `identity` and `user` core (name, persona, communication style) are never cut
2. **Conversation history** — drop oldest message pairs first (existing `trim_history()` behaviour, now budget-driven)
3. **RAG memories** — reduce from top-K to top-1, then drop entirely if still over
4. **Vision context** — truncate to a one-sentence summary, then drop
5. **Time section** — always the last to go; dropped only in extreme cases

#### Implementation

- **`src/llm/context.py`** — `ContextBudget` class: holds the tier percentages and token/char limits, exposes `assemble(soul, history, rag, vision, time) -> dict[str, str]` which returns each section at its trimmed size and a `was_trimmed: bool` flag per section
- **Token counting** — approximated as `len(text) // 4` (1 token ≈ 4 chars) for zero-dependency speed; a more accurate `tiktoken`-based counter can be swapped in later via a single interface
- **Soul trimmer** (`src/memory/soul.py`) — `to_prompt_section(budget_chars: int)` gains an optional budget parameter; if the full YAML exceeds the budget it progressively drops `facts` → `environment` → extended `user` fields until it fits
- **Logging** — whenever any section is trimmed, an entry is appended to `data/logs/context_trim.log` (built in Phase 1.8) recording which sections were cut, by how many tokens, and the final assembled size
- **Streamlit indicator** — if any trimming occurred on the last message, a subtle `⚠ context trimmed` badge appears next to the message count in the sidebar; clicking it opens the Context trim log tab
- **Tests**: `tests/test_context.py` — 10/10 passing

### ✅ Phase 1.9b — Memory Strategy & Intelligence *(complete)*

Decision logic governing *what* gets stored in short-term vs. long-term memory, so all later phases follow consistent rules.

- **Short-term memory policy** (`src/memory/policy.py`): `is_filler_message()` pre-filter excludes low-value turns (≤10 chars or exact filler phrases) from the history buffer; `compress_history()` in `src/llm/client.py` summarises oldest turn-pairs into a `[Earlier context summary]` entry on overflow rather than discarding them
- **Long-term memory policy** (`src/memory/policy.py`): `should_store()` gate enforces novelty, stability, category whitelist (`LONG_TERM_CATEGORIES`), and confidence threshold (`CONFIDENCE_THRESHOLD = 0.8`); `_is_transient()` keyword-pattern check rejects ephemeral facts (weather, current location, etc.)
- **Soul patch confidence**: `SOUL_PATCH_PROMPT` requires `_confidence` and `_explicit` metadata; `_patch_worker` strips them before `apply_patch()` and skips low-confidence non-explicit patches
- **Two-pass extraction** (`src/memory/extractor.py`): `maybe_extract_memories()` fires every 5 turns as a daemon thread — sends `MEMORY_EXTRACT_PROMPT` to extract `{fact, category, confidence, explicit}` candidates, runs each through `should_store()`, and logs all decisions to `data/logs/memory.log`
- **Curiosity system**: `maybe_grow_curiosity()` periodically asks the LLM to add questions to a `curiosity_queue` in the soul file; Orion asks them naturally when the moment fits
- **Tests**: `tests/test_memory_policy.py` — 25/25 passing

### ✅ Phase 2 — Persistent Memory (RAG) *(complete)*

Orion has long-term episodic memory that persists across conversations. On each user message the relevant memory is retrieved and injected into context; sessions are summarised and stored when a conversation ends.

- **Vector Database**: [ChromaDB](https://github.com/chroma-core/chroma) — lightweight, embedded, fully local, persists to `data/memory/`
- **Embeddings**: [sentence-transformers](https://www.sbert.net/) `all-MiniLM-L6-v2` — 384-dim, CPU-only, 22 MB, module-level cache so multiple instances share one load
- **`src/memory/vector_store.py`** — `MemoryStore` with `add_memory()` (SHA-256 ID, idempotent upsert), `query_memory()` (cosine similarity threshold ≥ 0.35, returns `[]` when nothing qualifies), `clear_memory()`, `count()`; every call logged to `memory.log`
- **`src/memory/embeddings.py`** — `Embedder` class wrapping sentence-transformers; `embed()` and `embed_batch()` with empty-string validation
- **`src/memory/summariser.py`** — `summarise_session()` feeds the full conversation to the LLM with `SUMMARISE_SESSION_PROMPT`; returns `""` when fewer than 2 user turns or when the LLM produces only whitespace
- **RAG injection** (`src/app.py`, `src/main.py`): on each non-filler user message, query the store → if results clear the threshold, build a `## Relevant Memory` block capped at `ContextBudget.rag_budget_chars()` and append to the system prompt; skip entirely when nothing is relevant
- **Session save**: `app.py` saves a summary when "🗑️ Clear conversation" is clicked; `main.py` saves in a `try/finally` so both clean `quit` and Ctrl-C are covered
- **Tests**: `tests/test_memory.py` 14/14 + `tests/test_embeddings.py` 15/15 + `tests/test_summariser.py` 19/19 + `tests/test_rag_integration.py` 17/17 = **65 tests passing**

### Phase 3 — Voice Input (Speech-to-Text)

Transcribe speech from a USB microphone in near real-time using a local Whisper model.

- **Library**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 reimplementation of OpenAI Whisper, up to 4× faster with lower VRAM usage
- **Model**: `whisper small.en` (~2GB VRAM on GPU) for accuracy; `tiny.en` (~1GB) for speed
- **Audio capture**: `sounddevice` + `numpy` for microphone streaming; VAD (Voice Activity Detection) via the built-in Silero VAD filter to clip silence automatically
- **Flow**: Microphone stream → VAD filter → Whisper transcription → text passed to LLM pipeline

### Phase 4 — Voice Output (Text-to-Speech)

Have the robot speak responses aloud through a speaker in a natural-sounding voice, with minimal latency.

- **Primary**: [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) — fast, neural, offline TTS that runs on CPU (great for Raspberry Pi too); supports many voice models
- **Alternative**: [Kokoro TTS](https://github.com/hexgrad/kokoro) — high-quality, lightweight, Python-native, recently released (Apache 2.0)
- **Audio playback**: `sounddevice` or `pygame.mixer`
- **Flow**: LLM response text → sentence boundary split → TTS model → audio playback (streaming, sentence-by-sentence for low latency)

### Phase 5 — Vision

Use a USB camera to capture the robot's surroundings, identify objects and scenes, and feed that context to the LLM.

- **Object Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — state-of-the-art, real-time detection, runs on CUDA; produces a list of detected objects + confidence scores
- **Scene Understanding / VLM**: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) — a full vision-language model that can both describe a scene in natural language and answer questions about it; available via `ollama` as `qwen2.5vl:7b` (~5GB 4-bit)
- **Camera capture**: `opencv-python` (`cv2`) for frame capture and preprocessing
- **Lightweight alternative for Pi**: [moondream2](https://huggingface.co/vikhyatk/moondream2) — 1.8B VLM optimised for edge devices
- **Flow**: Camera frame → YOLOv8 detection list + optional VLM scene caption → injected into LLM context window as a "vision summary"

### Phase 5.5 — Face Recognition

Teach Orion to detect and identify people he sees — greeting Daniel and Danielle by name, and asking who an unrecognised person is so he can learn them.

- **Library**: [`face_recognition`](https://github.com/ageitgey/face_recognition) (dlib-backed) — CPU-only, works on both laptop and Raspberry Pi ARM64
- **Enrollment**: capture reference photos of known people and store 128-dimensional face encodings locally in `data/faces/`; manual enrollment via a CLI helper script
- **Recognition flow**: on each camera frame, detect faces → compare encodings against enrolled entries with a Euclidean distance threshold → return the best match (or `"unknown"` if none clears the threshold)
- **LLM integration**: inject recognised identities silently under a `## People Present` section in the LLM context so Orion can refer to them naturally; on first recognition in a session, greet them by name aloud (via TTS)
- **Unknown face handling**: when an unrecognised face is detected, Orion asks who the person is and optionally enrolls them with their permission — the new encoding is saved to `data/faces/` and the soul file `user` section is updated
- **Pi-compatible**: dlib and `face_recognition` build cleanly on Raspberry Pi OS 64-bit (aarch64) with a pre-compiled wheel; inference stays on CPU alongside the LLM GPU workload on the laptop

### Phase 5.6 — Network Bridge (Laptop Brain / Pi Body)

Run all AI inference on the laptop while the Raspberry Pi acts as a thin I/O client over local Wi-Fi — no heavy models on the Pi for this phase.

- **Design**: laptop runs a [FastAPI](https://fastapi.tiangolo.com/) bridge server; Pi connects as a WebSocket/HTTP client, streams raw sensor data in, receives processed results and commands back
- **Audio pipeline**: Pi captures raw PCM audio from USB microphone → streams 16 kHz chunks to laptop over WebSocket → laptop runs `faster-whisper` (GPU) with Silero VAD → transcribed text returned to Pi
- **Vision pipeline**: Pi captures JPEG frames → HTTP POST to laptop `/vision` endpoint → laptop runs YOLO + optional VLM → returns a vision summary string
- **LLM + memory**: transcribed text enters the full laptop-side pipeline (soul file, RAG, context budget, Ollama) → response streamed back to Pi token-by-token over WebSocket
- **TTS on Pi**: laptop sends the final response text only; Pi synthesises speech locally with Piper TTS (avoids streaming large audio files over Wi-Fi)
- **Motor commands**: if the LLM output includes a `tool_call` JSON block, the laptop embeds it in the final WebSocket message; Pi parses and executes it via `gpiozero`
- **Service discovery**: Pi auto-discovers the laptop server using mDNS (`zeroconf`) — no hardcoded IP addresses; falls back to `ORION_SERVER_URL` environment variable
- **Security**: a shared auth token (stored in `.env`) is sent as a Bearer header on every request so the server is not open to the entire LAN

### Phase 6 — Physical Robot (Raspberry Pi)

Package everything as a self-contained robot with motors, running on a Raspberry Pi 5.

- **Hardware**:
  - Raspberry Pi 5 (8GB RAM recommended)
  - 2WD or 4WD robot chassis kit with DC motors
  - L298N dual H-bridge motor driver module
  - USB microphone + small USB speaker
  - Raspberry Pi Camera Module v3 (or USB camera)
  - LiPo battery pack or USB-C power bank (for portability)
- **LLM on Pi**: [llama.cpp](https://github.com/ggerganov/llama.cpp) with a quantised `Q4_K_M` version of Phi-3 Mini 3.8B or TinyLlama 1.1B
- **STT on Pi**: `faster-whisper` with `tiny.en` model (CPU-only on Pi)
- **TTS on Pi**: Piper TTS (C++ binary, very low resource usage)
- **Motor Control**: `gpiozero` library with L298N driver; simple movement commands (forward, back, left, right, stop) issued by the LLM via a simple tool-call interface
- **Offload option**: Pi handles audio I/O and motor control; heavy inference (LLM + vision) offloaded to laptop over local network

---

## Hardware Requirements

### Laptop (Development)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 4060 (8GB VRAM) | Same |
| RAM | 16GB | 32GB |
| Storage | 50GB free | 100GB+ SSD |
| OS | Windows 10/11 | Windows 11 |
| CUDA | 12.0+ | 12.4+ |
| Python | 3.10 | 3.11 |

### Peripherals (All Phases)

| Device | Purpose | Notes |
|--------|---------|-------|
| USB Microphone | Voice input | Any standard USB mic works |
| USB Speaker / 3.5mm speaker | Voice output | USB preferred for simplicity |
| USB Webcam | Vision | 720p+ recommended |

### Raspberry Pi Robot (Phase 6)

| Component | Details |
|-----------|---------|
| Raspberry Pi 5 | 8GB RAM variant recommended |
| Robot chassis | 2WD or 4WD kit with DC motors |
| Motor driver | L298N dual H-bridge module |
| Power | 5V/5A USB-C for Pi; separate battery for motors |
| Camera | Raspberry Pi Camera Module v3 or USB webcam |
| Microphone | USB or I2S MEMS microphone |
| Speaker | USB powered or 3.5mm with amp |

---

## Software Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| LLM inference | [Ollama](https://ollama.com/) + [Phi-4-mini](https://huggingface.co/microsoft/phi-4-mini-instruct) | Easy model management, CUDA acceleration, no API key |
| LLM framework | [llama.cpp](https://github.com/ggerganov/llama.cpp) | Low-level GGUF inference on Pi |
| Memory | [ChromaDB](https://github.com/chroma-core/chroma) | Embedded vector DB, persists to disk |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) | Fast, small, accurate |
| Speech-to-Text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 4× faster than stock Whisper, CUDA/CPU |
| Text-to-Speech | [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) / [Kokoro](https://github.com/hexgrad/kokoro) | Fast, local, neural voices |
| Vision (detection) | [YOLOv8](https://github.com/ultralytics/ultralytics) | Real-time object detection |
| Vision (VLM) | [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Scene description + visual Q&A |
| Camera capture | [OpenCV](https://opencv.org/) | Cross-platform, well-supported |
| Face recognition | [`face_recognition`](https://github.com/ageitgey/face_recognition) (dlib) | CPU-only, Pi-compatible, simple enrollment API |
| Audio I/O | [sounddevice](https://python-sounddevice.readthedocs.io/) | Cross-platform microphone/speaker |
| Motor control | [gpiozero](https://gpiozero.readthedocs.io/) | Raspberry Pi GPIO abstraction |
| Network bridge | [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/) | Async HTTP + WebSocket server; laptop-to-Pi data channel (Phase 5.6) |
| Service discovery | [zeroconf](https://github.com/python-zeroconf/python-zeroconf) | mDNS auto-discovery so Pi finds the laptop without a hardcoded IP (Phase 5.6) |
| Language | Python 3.11 | Universal support across all libraries |

---

## Models

All models run locally — no API keys, no internet required at inference time.

### Language Models (LLM)

| Model | Parameters | Quant | Size | Context Window | Hardware | Docs |
|---|---|---|---|---|---|---|
| [Phi-4-mini-instruct](https://huggingface.co/microsoft/phi-4-mini-instruct) *(primary)* | 3.8B | Q4_K_M | 2.5 GB | 128K tokens | RTX 4060 (GPU) | [Model card](https://huggingface.co/microsoft/phi-4-mini-instruct) · [Technical report](https://arxiv.org/abs/2503.01743) |
| [Llama 3.2 3B](https://ollama.com/library/llama3.2) *(alternative)* | 3B | Q4_K_M | 2.0 GB | 128K tokens | RTX 4060 (GPU) | [Model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) · [Meta blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) |
| [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) *(Pi)* | 3.8B | Q4_K_M | 2.2 GB | 4K tokens | Raspberry Pi 5 (CPU) | [Model card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) *(Pi fallback)* | 1.1B | Q4_K_M | ~0.7 GB | 2K tokens | Raspberry Pi 5 (CPU) | [Model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) · [Paper](https://arxiv.org/abs/2401.02385) |

> **Active model** on this machine: `phi4-mini` via Ollama. Context window is intentionally capped at 4K chars (~1K tokens) in code to keep GPU VRAM usage predictable; the model's full 128K window is available if needed.

### Embeddings

| Model | Dimensions | Max Input | Size | Hardware | Docs |
|---|---|---|---|---|---|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | 256 tokens | 22 MB | CPU | [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [SBERT docs](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) |

### Speech-to-Text (Whisper)

| Model | Size | VRAM | WER (en) | Hardware | Docs |
|---|---|---|---|---|---|
| [faster-whisper small.en](https://huggingface.co/Systran/faster-whisper-small.en) *(laptop)* | ~244 MB | ~2 GB | ~4% | RTX 4060 (GPU) | [Repo](https://github.com/SYSTRAN/faster-whisper) · [Model card](https://huggingface.co/Systran/faster-whisper-small.en) |
| [faster-whisper tiny.en](https://huggingface.co/Systran/faster-whisper-tiny.en) *(Pi)* | ~39 MB | CPU only | ~6% | Raspberry Pi 5 (CPU) | [Model card](https://huggingface.co/Systran/faster-whisper-tiny.en) |

> Whisper processes audio in 30-second chunks; there is no token context window in the traditional sense.

### Text-to-Speech

| Model | Type | Latency | Hardware | Docs |
|---|---|---|---|---|
| [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) *(primary)* | Neural VITS | Very low | CPU (laptop + Pi) | [Repo](https://github.com/OHF-Voice/piper1-gpl) · [Voice list](https://github.com/rhasspy/piper/blob/master/VOICES.md) |
| [Kokoro TTS](https://github.com/hexgrad/kokoro) *(alternative)* | StyleTTS2-based | Low | CPU / GPU | [Repo](https://github.com/hexgrad/kokoro) · [HF space](https://huggingface.co/hexgrad/Kokoro-82M) |

### Vision

| Model | Task | Size | Context / Input | Hardware | Docs |
|---|---|---|---|---|---|
| [YOLOv8n](https://docs.ultralytics.com/models/yolov8/) | Object detection | ~6 MB | Single frame | RTX 4060 (GPU) | [Docs](https://docs.ultralytics.com/models/yolov8/) · [Paper](https://arxiv.org/abs/2305.09972) |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Vision-language (VLM) | ~5 GB (4-bit) | 32K tokens + image | RTX 4060 (GPU) | [Model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) · [Paper](https://arxiv.org/abs/2502.13923) |
| [moondream2](https://huggingface.co/vikhyatk/moondream2) | Vision-language (Pi) | ~1.8 GB | 2K tokens + image | Raspberry Pi 5 (CPU) | [Model card](https://huggingface.co/vikhyatk/moondream2) · [Repo](https://github.com/vikhyatk/moondream) |

---

## Project Structure

```
personal_robot/
├── README.md
├── TODO.md
├── requirements.txt          # Laptop requirements
├── requirements_pi.txt       # Raspberry Pi requirements (lighter)
├── src/
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py         # Ollama / llama.cpp wrapper
│   │   ├── context.py        # ContextBudget — percentage-based context assembly
│   │   └── prompts.py        # System prompt templates
│   ├── utils/
│   │   ├── __init__.py
│   │   └── log.py            # Central logging factory (get_logger)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── soul.py           # SoulFile — identity YAML, LLM-driven self-update
│   │   ├── policy.py         # Memory policy — should_store(), filler filter
│   │   ├── extractor.py      # Two-pass memory extraction daemon
│   │   ├── vector_store.py   # ChromaDB read/write with memory.log logging
│   │   ├── embeddings.py     # sentence-transformers Embedder wrapper
│   │   └── summariser.py     # LLM-powered session summariser
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── stt.py            # faster-whisper microphone transcription
│   │   └── tts.py            # Piper / Kokoro speech synthesis
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera.py         # OpenCV frame capture
│   │   ├── detector.py       # YOLOv8 object detection
│   │   ├── vlm.py            # Vision-language model (Qwen2.5-VL)
│   │   └── faces.py          # face_recognition enroll + identify
│   ├── robot/
│   │   ├── __init__.py
│   │   └── motors.py         # gpiozero motor control (Pi only)
│   ├── app.py                # Streamlit chat UI
│   ├── main.py               # CLI conversation loop
│   └── server.py             # FastAPI bridge server — laptop-side (Phase 5.6)
├── data/
│   ├── logs/                     # All runtime logs (gitignored)
│   │   ├── conversations/        # Per-session JSONL turn logs
│   │   ├── soul_changes.log      # Soul file patch diffs
│   │   ├── context_trim.log      # Short-term memory trim events
│   │   └── memory.log            # RAG retrieval + write events (Phase 2)
│   └── memory/               # Persistent ChromaDB files (gitignored)
├── models/                   # Downloaded GGUF / ONNX models (gitignored)
├── raspberry_pi/
│   ├── README.md             # Pi-specific setup guide
│   ├── setup.sh              # Pi setup script
│   ├── main_pi.py            # Lightweight Pi entry point (standalone, Phase 6)
│   └── bridge_client.py      # Thin I/O client — connects to laptop server (Phase 5.6)
└── tests/
    ├── test_llm.py
    ├── test_soul.py
    ├── test_context.py
    ├── test_logging.py
    ├── test_memory.py
    ├── test_memory_policy.py
    ├── test_embeddings.py
    ├── test_summariser.py
    ├── test_rag_integration.py
    ├── test_audio.py           # Phase 3 (not yet implemented)
    └── test_vision.py          # Phase 5 (not yet implemented)
```

---

## Getting Started

### Prerequisites

1. Install [Ollama](https://ollama.com/download) for Windows
2. Pull the Phi-4-mini model:
   ```bash
   ollama pull phi4-mini
   ```
3. Install CUDA Toolkit 12.x from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
4. Create a Python virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Run the CLI conversation loop

```bash
.venv\Scripts\python.exe src/main.py
```

### Run the Streamlit chat UI

```bash
.venv\Scripts\streamlit.exe run src/app.py
```

Opens at `http://localhost:8501`. Use the sidebar to switch models, edit the system prompt, or clear the conversation.

### View runtime logs

All logs are written to `data/logs/` (gitignored). The most useful files:

| Log file | What it records |
|---|---|
| `conversations/<timestamp>.jsonl` | Every conversation turn — role, content, model, timestamp |
| `soul_changes.log` | Each soul file patch — before/after diff |
| `context_trim.log` | Messages dropped when the context window is trimmed |
| `memory.log` | RAG queries + similarity scores, session summary writes (Phase 2+) |

The Streamlit sidebar (`src/app.py`) includes a tabbed log viewer for all four files. For the CLI, open any log directly in a text editor or tail it:

```bash
Get-Content data\logs\soul_changes.log -Wait   # PowerShell live tail
```

### Run the test suite

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Raspberry Pi Deployment

See [raspberry_pi/README.md](raspberry_pi/README.md) for a dedicated setup guide covering:

- Operating system setup (Raspberry Pi OS 64-bit)
- llama.cpp compilation with optimisations
- Piper TTS binary installation
- GPIO wiring diagram for the L298N motor driver
- Auto-start configuration via `systemd`

---

## Licence

MIT — see [LICENSE](LICENSE) for details.

---

## Exploratory Research & Design

### Thinking Fast and Slow

“Thinking, Fast and Slow” by Daniel Kahneman maps human cognition onto two systems, and they translate surprisingly well into LLM memory architecture.

#### The Core Mapping

**System 1 → Fast, implicit, low-cost inference**  
Kahneman’s System 1 is automatic, associative, and effortless. In LLM terms, this maps to:

- The base model’s parametric memory — knowledge baked into weights during training
- Embedding-based retrieval (semantic search over a vector store) — fast, fuzzy, pattern-matching recall
- KV cache reuse within a context window — instant, zero-compute recall of recent tokens

**System 2 → Slow, deliberate, high-cost reasoning**  
System 2 is effortful, rule-based, and sequential. In LLM terms:

- Chain-of-thought / scratchpad reasoning — the model “thinking step by step”
- Tool-augmented retrieval — deliberately querying structured knowledge bases, APIs, or long-term stores
- Agentic loops — multi-step planning where the model reflects, verifies, and revises

#### Key Insights from the Book and Their Architectural Implications

1. **Cognitive ease drives System 1 dominance**  
   Kahneman shows that familiar, fluent inputs trigger System 1 by default. For LLMs, this suggests a router/gating mechanism: route common, well-formed queries to fast parametric or cached retrieval, and only escalate to expensive retrieval or CoT when novelty or ambiguity is detected. This is essentially what systems like Mixtral’s MoE or speculative decoding approximate at the compute level.
2. **Availability heuristic → Recency bias in memory**  
   System 1 over-weights recent, vivid memories. An honest LLM memory system should model this explicitly — giving recency-weighted scores in retrieval, but also building in a System 2 correction pass that asks “is this the most relevant memory, or just the most recent one?”
3. **Anchoring → Prompt/context poisoning**  
   Early information in a context window disproportionately anchors later reasoning, just as anchoring biases System 1. Architecturally, this argues for position-agnostic retrieval (don’t just stuff retrieved memory at the top of the prompt) and explicit conflict-detection when retrieved facts contradict the anchor.
4. **What You See Is All There Is (WYSIATI)**  
   This is Kahneman’s most important insight for AI: System 1 builds a coherent narrative from only the information present, without flagging what’s missing. LLMs do this acutely. The architectural remedy is a memory completeness check — a System 2 step that asks “what information would I need but don’t have?” before committing to an answer. This is the spirit behind techniques like self-RAG and uncertainty-aware generation.
5. **Dual-process for memory consolidation**  
   Kahneman describes sleep and reflection as when System 2 reviews and consolidates System 1 experiences. You can mirror this with an asynchronous consolidation loop — a background process that periodically summarizes, deduplicates, and re-indexes episodic memory into a more compressed semantic store. This is exactly what systems like MemGPT attempt.

#### A Rough Architecture Sketch

```text
Incoming query
      │
      ▼
 ┌─────────────────────────────────┐
 │  SYSTEM 1 LAYER (fast path)     │
 │  • KV cache (in-context)        │
 │  • Vector similarity retrieval  │
 │  • Parametric weights           │
 │  Confidence score emitted ──────┼──► High confidence → respond
 └─────────────────────────────────┘
             │ Low confidence / novelty detected
             ▼
 ┌─────────────────────────────────┐
 │  SYSTEM 2 LAYER (slow path)     │
 │  • Structured KB / SQL queries  │
 │  • CoT / scratchpad reasoning   │
 │  • Conflict detection           │
 │  • WYSIATI completeness check   │
 └─────────────────────────────────┘
             │
             ▼
      Final response
             │
    (Async, background)
             ▼
 ┌─────────────────────────────────┐
 │  CONSOLIDATION (sleep loop)     │
 │  • Summarize episodic buffer    │
 │  • Merge into semantic store    │
 │  • Prune redundant memories     │
 └─────────────────────────────────┘
```

#### Where This Gets Hard

- The gating problem: Deciding when to escalate from System 1 to System 2 is itself a hard inference problem. Getting this wrong is costly either way (too slow, or confidently wrong).
- Metacognition is weak in LLMs: System 2 in humans involves genuine awareness of System 1’s errors. LLMs lack reliable introspection about their own confidence, making the handoff noisy.
- Bias doesn’t disappear: Kahneman’s point is that System 2 is lazy — it often just endorses System 1’s output. LLMs exhibit exactly this; CoT doesn’t always catch parametric errors, it sometimes just rationalizes them.
