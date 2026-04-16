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

- [ ] Define a YAML schema with the following top-level sections:
  ```yaml
  identity:        # Robot's name, personality traits, capabilities
  user:            # User's name, preferences, facts learned about them
  environment:     # Location, connected hardware, known surroundings
  facts:           # General facts the robot has learned and wants to retain
  ```
- [ ] Create a default `data/soul.yaml` seeded with the robot's base identity (name, phase capabilities, hardware)
- [ ] Add `data/soul.yaml` to `.gitignore` so personal facts are not committed

### 1.7.2 — Soul File Module (`src/memory/soul.py`)

- [ ] Create a `SoulFile` class that reads and writes `data/soul.yaml`
- [ ] Implement `load() -> dict` — parses YAML and returns the full soul dict
- [ ] Implement `to_prompt_section() -> str` — formats the soul as a `## About Me` block for injection into the system prompt
- [ ] Implement `update(section: str, key: str, value: str)` — writes a single fact to the correct section and saves the file
- [ ] Implement `apply_patch(patch: dict)` — bulk-applies a dict of `{section: {key: value}}` updates from the LLM
- [ ] Add file locking so concurrent writes (e.g., from Streamlit) don't corrupt the file

### 1.7.3 — LLM-Driven Updates

- [ ] Add a post-response step that prompts the LLM: *"Did you learn any new facts worth remembering? If so, output a JSON patch block, otherwise output nothing."*
- [ ] Parse the LLM's response for a fenced `json` block shaped like `{"section": {"key": "value"}}`
- [ ] Call `soul.apply_patch(patch)` if a valid patch is found; silently skip if the LLM outputs nothing
- [ ] Ensure the update prompt only runs every N turns (e.g., every 5 messages) to avoid overhead

### 1.7.4 — Integrate Soul File into Conversation Loop

- [ ] On startup in `src/main.py`: load soul file and inject `soul.to_prompt_section()` into the system prompt
- [ ] On startup in `src/app.py`: same injection, reload soul on each new session
- [ ] Add a "Soul file" expander in the Streamlit sidebar showing the current YAML contents (read-only view)
- [ ] Write `tests/test_soul.py` — test load, update, apply_patch, and prompt injection

---

## Phase 2 — Persistent Memory (RAG)

**Goal**: The LLM remembers information from previous conversations by retrieving relevant past interactions before each response.

### 2.1 — Set Up ChromaDB (`src/memory/vector_store.py`)

- [ ] Add `chromadb` and `sentence-transformers` to `requirements.txt`
- [ ] Create a `MemoryStore` class that wraps ChromaDB with persistence to `data/memory/`
- [ ] Implement `add_memory(text: str, metadata: dict)` — embeds text and stores it
- [ ] Implement `query_memory(text: str, n_results: int = 5) -> list[str]` — retrieves top-K relevant memories
- [ ] Implement `clear_memory()` for development/testing

### 2.2 — Embeddings (`src/memory/embeddings.py`)

- [ ] Download and cache `all-MiniLM-L6-v2` model from `sentence-transformers`
- [ ] Create an `Embedder` class with an `embed(text: str) -> list[float]` method
- [ ] Confirm embeddings are generated on CPU (reserve GPU for LLM inference)

### 2.3 — Session Summariser (`src/memory/summariser.py`)

- [ ] Write a `summarise_session(conversation_history: list[dict]) -> str` function
- [ ] Feed the full conversation to the LLM with a summarisation prompt
- [ ] Return a concise paragraph capturing key facts, preferences, or information mentioned

### 2.4 — Integrate Memory into Conversation Loop

- [ ] On each user message: query `MemoryStore` for top-5 relevant past memories
- [ ] Inject retrieved memories into the system prompt as a `## Relevant Memory` section
- [ ] On conversation end (graceful exit): call `summarise_session()` and save the summary to `MemoryStore`
- [ ] Write `tests/test_memory.py` to confirm persistence across two separate Python runs

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
