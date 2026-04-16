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

### Phase 1.7 — Soul File (Identity & Long-Term Facts)

Give the LLM a persistent, human-readable YAML file that defines its identity and stores facts it learns over time.

- **File**: `data/soul.yaml` — sections for `identity`, `user`, `environment`, and `facts`; gitignored so personal data stays local
- **Module**: `src/memory/soul.py` — `SoulFile` class with `load()`, `to_prompt_section()`, `update()`, and `apply_patch()`
- **Self-updating**: after every N turns, the LLM is asked whether it learned anything worth keeping; if it outputs a JSON patch block, it is automatically written back to the soul file
- **Injection**: the soul file is loaded on startup and injected into the system prompt as a `## About Me` section, so the LLM always has its identity and known facts in context
- **Scope — identity layer only**: the soul file is the *always-on* layer injected into every prompt. It should stay compact and contain only stable, perpetually-relevant facts (Orion's identity, Daniel's core profile). Episodic knowledge — what was discussed in past sessions, situational preferences, one-off facts — belongs in Phase 2 RAG, not in the soul file's `facts` section.
- **Distinct from RAG**: the soul file is structured and hand-editable; Phase 2 RAG handles unstructured episodic search over conversation history

### Phase 2 — Persistent Memory (RAG)

Give the robot long-term episodic memory that persists across conversations. This is the *selective* long-term memory layer — Orion only draws on it when the current conversation makes it relevant. The in-conversation rolling message list already serves as short-term memory; this phase adds cross-session retrieval on top.

- **Vector Database**: [ChromaDB](https://github.com/chroma-core/chroma) — lightweight, embedded, fully local, persists to disk
- **Embeddings**: [sentence-transformers](https://www.sbert.net/) with `all-MiniLM-L6-v2` — fast, small (22MB), runs on CPU
- **Memory Flow**:
  1. On each user message: embed the message and query the vector store for the top-K most similar past session summaries
  2. Apply a minimum cosine similarity threshold (e.g. ≥ 0.35) — if no result clears the threshold, skip injection entirely; Orion does not reach into long-term memory unless it is genuinely relevant
  3. Inject qualifying memories into the LLM context under a `## Relevant Memory` section
  4. At conversation end (graceful exit): use the LLM to summarise the session and save the summary to the vector store
- **Conversation Summarisation**: Use the LLM itself to summarise each session before saving

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
| Language | Python 3.11 | Universal support across all libraries |

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
│   │   └── prompts.py        # System prompt templates
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── vector_store.py   # ChromaDB read/write
│   │   ├── embeddings.py     # sentence-transformers wrapper
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
│   └── main.py               # CLI conversation loop
├── data/
│   └── memory/               # Persistent ChromaDB files (gitignored)
├── models/                   # Downloaded GGUF / ONNX models (gitignored)
├── raspberry_pi/
│   ├── README.md             # Pi-specific setup guide
│   ├── setup.sh              # Pi setup script
│   └── main_pi.py            # Lightweight Pi entry point
└── tests/
    ├── test_llm.py
    ├── test_memory.py
    ├── test_audio.py
    └── test_vision.py
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
