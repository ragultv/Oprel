# Changelog

All notable changes to the Oprel SDK and runtime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.7.2] - 2026-11-09

### 🎙️ Audio Studio & Realtime Speech Engine
- **Kokoro-82M Text-to-Speech (TTS)**:
  - Local GPU-accelerated speech synthesis with CUDA acceleration.
  - High-capacity buffer supporting up to 100,000 characters in a single job.
  - Multi-voice support with American and British regional accents (`Heart`, `Sarah`, `Bella`, `Adam`, `Michael`, `Emma`, `George`).
  - Audio normalization and playback speed scaling (0.5x - 2.0x).
- **Faster-Whisper Speech-to-Text (STT)**:
  - Real-time live microphone recording with streaming chunk recognition.
  - Multi-format file audio transcription (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.webm`).
  - Warmup daemon and background pooling for zero-latency cold-start transcription.
- **Waveform Audio Player**:
  - Interactive scrub bar, speed multiplier, playback controls, and one-click `.wav` audio download.

---

### 📚 Knowledge Studio & Universal Document Engine
- **Universal Multi-Format Document Viewer**:
  - **Visual PDF Viewer**: Embedded native PDF reader with zoom, multi-page scroll, and print controls, served via non-cached `Content-Disposition: inline` streams.
  - **Tabular Data Grid**: Interactive CSV and TSV preview with sticky headers, zebra striping, and cell search filtering.
  - **GitHub-Flavored Markdown**: Full rendered view with code blocks, tables, lists, and raw source toggle.
  - **Source Code Viewer**: Line-numbered monospace syntax container for Python, TypeScript, JavaScript, HTML, CSS, SQL, Shell, JSON, and YAML.
  - **Document Reader Layout**: Formatted text reader with real-time character/word counts, line statistics, and estimated reading times.
- **Recursive Semantic Chunker (`SemanticChunker`)**:
  - Hierarchical splitting respecting Markdown headers (`##`, `###`), paragraphs (`\n\n`), and sentence boundaries (`. `).
  - Target chunk capacity of 400 characters with 80-character rolling overlap and heading context inheritance.
- **Hybrid Retrieval & RRF Fusion**:
  - Reciprocal Rank Fusion combining dense semantic embeddings (ChromaDB / HNSW) with sparse lexical search (SQLite FTS / BM25).
  - Context deduplication and reranking for high-precision context injection.
- **Multi-Directory Ingestion & Auto-Discovery**:
  - Unified document scanning resolving across `knowledge_files`, `my_docs`, and `tmp_uploads`.

---

### 👁️ OCR Studio & Vision Language Model (VLM)
- **Dots OCR Engine (`dots.ocr-GGUF`)**:
  - High-precision document extraction utilizing multi-modal vision projection layers (`mmproj-dots.ocr`).
- **5-View Extraction Workbench**:
  - Tabbed results viewer: **Text**, **Markdown**, **Table**, **Lines**, and raw **JSON**.
  - Interactive document canvas with zoom, rotate, pan, and bounding box inspection.
- **OCR Settings & History**:
  - Real-time confidence threshold slider, bounding box toggle, and structured table toggle.
  - Filterable history drawer with timestamp filters and one-click session reloading.

---

### 🎨 Image Studio Redesign
- **Unified Studio Architecture**:
  - Full-width sticky `h-14` header docked above workspace and settings sidebar, matching Audio, OCR, and Knowledge Studios.
  - Docked **Settings** and **History** sidebars with horizontal slider controls and active model highlight cards.
- **Generation Controls**:
  - Aspect ratio presets (`1:1 Square`, `16:9 Landscape`, `9:16 Portrait`, `4:3 Classic`), inference steps (5-40), CFG guidance scale (1.0-15.0), and sampling methods (Euler, Euler A, DPMPP 2M, etc.).
  - Inspiration prompt cards and random surprise prompt generator.

---

### ⚡ Runtime, KV Cache & Memory Optimization
- **Quantized KV Cache (`Q4_0` & `Q8_0`)**:
  - Cuts KV cache VRAM footprint by up to 50% on long context sequences.
- **Intelligent Context Compactor (`context_compactor.py`)**:
  - Token reserve estimation using `cl100k_base` / `tiktoken` with middle-truncation preserving essential prefix and suffix context.
- **Dynamic VRAM Budgeting & Layer Offloader**:
  - Auto-calculates optimal GPU layer offloading based on detected NVIDIA VRAM and system RAM.
  - Graceful fallback to CPU quantization when VRAM thresholds are saturated.
- **Daemon Lifecycle Management**:
  - Automatic orphaned backend process detection and cleanup on daemon start/stop.

---

### 🛡️ Security Hardening & Authentication
- **Hugging Face Token Integration**:
  - Dedicated authentication modal (`HfConnectModal.tsx`) and verification routes for private/gated Hugging Face model downloads.
- **LocalGuard Security Middleware**:
  - Protection against DNS rebinding and cross-site script execution.
  - Host validation restricting traffic to loopback interfaces (`127.0.0.1`, `localhost`).
  - Bearer token verification (`~/.oprel/api_token`) for external SDK/CLI client access.
- **Encrypted Local Secrets Store**:
  - Secure credential persistence for API keys and tokens (`secrets_store.py`).

---

### 🖥️ Unified WebUI & Design System
- **Studio Shell Geometry**:
  - Unified header geometry across Chat, Audio, Image, OCR, and Knowledge views.
  - Resizable AppShell with persistent sidebar width and drag dividers.

---

## [0.7.1] - 2026-10-14
- Published initial version 0.7.1 with security hardening documentation.
- Initialized core server and version modules.
- Multi-provider support with dynamic model listing and OpenAI-compatible routing.

---

## [0.7.0] - 2026-10-08
- Implemented FastAPI-based background server daemon (`oprel start`).
- Added installation hardening and binary integrity manifest foundation.
- Added support for local model pulls, hardware diagnostics, and telemetry.

---

## [0.6.2] - 2026-08-10
- Baseline runtime library release.
- Added model parameter and quantization analysis utilities.
- Hardware-aware deployment guidelines and documentation.

