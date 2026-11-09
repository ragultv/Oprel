from __future__ import annotations

import gc
import io
import math
import os
import re
import shutil
import struct
import sys
import threading
import time as time_module
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

if sys.platform == "win32":
    try:
        _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(_torch_lib)
            if _torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = _torch_lib + ";" + os.environ.get("PATH", "")
    except Exception:
        pass

from oprel.server.domain.state import get_state
from oprel.server.services.context import CONFIG, logger, synchronized_backend_operation
from oprel.server.services.model_state import force_unload_model


def get_speech_models_dir() -> Path:
    """
    Returns the root directory where speech models are stored.
    Uses the unified SDK cache directory (CONFIG.cache_dir) across all platforms.
    """
    cache_dir = CONFIG.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


AVAILABLE_TAGS = []


# Singleton model cache to avoid reloading weights repeatedly
_kokoro_engine: Optional[Any] = None
_whisper_engine: Optional[Any] = None
_whisper_loading: bool = False
_engine_lock = threading.Lock()


def _ensure_exclusive_backend() -> None:
    """
    Unloads any active llama.cpp / text LLM models to free up 100% of GPU VRAM
    before speech synthesis begins, exactly following the Stable Diffusion pattern.
    """
    state = get_state()
    for model_id, model in list(state.models.items()):
        logger.info("Unloading active model before speech generation: %s", model_id)
        force_unload_model(model_id, model)
        state.models.pop(model_id, None)
        state.model_configs.pop(model_id, None)
        state.model_last_used.pop(model_id, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_kokoro_installed_cache: Optional[bool] = None
_whisper_installed_cache: Optional[bool] = None


def invalidate_speech_models_cache():
    global _kokoro_installed_cache, _whisper_installed_cache
    _kokoro_installed_cache = None
    _whisper_installed_cache = None


def check_kokoro_installed() -> bool:
    global _kokoro_installed_cache
    if _kokoro_installed_cache is not None:
        return _kokoro_installed_cache
    models_dir = get_speech_models_dir() / "kokoro-82m"
    onnx_file = models_dir / "kokoro-v1.0.onnx"
    voices_file = models_dir / "voices-v1.0.bin"
    if onnx_file.exists() and voices_file.exists():
        _kokoro_installed_cache = True
        return True
    legacy_dir = Path("D:/Ragul/oprel-SDK/.speech_models/kokoro-82m")
    if (legacy_dir / "kokoro-v1.0.onnx").exists() and (legacy_dir / "voices-v1.0.bin").exists():
        _kokoro_installed_cache = True
        return True
    _kokoro_installed_cache = False
    return False



def check_whisper_installed() -> bool:
    global _whisper_installed_cache
    if _whisper_installed_cache is not None:
        return _whisper_installed_cache
    try:
        import faster_whisper
        _whisper_installed_cache = True
    except ImportError:
        _whisper_installed_cache = False
    return _whisper_installed_cache


def get_speech_models_status(include_all: bool = False) -> List[Dict[str, Any]]:
    """
    Returns the list and download status of speech models for the Audio Studio.
    """
    all_models = [
        {
            "id": "kokoro-82m",
            "name": "Kokoro-82M",
            "type": "tts",
            "engine": "kokoro",
            "description": "Ultra-fast speech synthesis with CUDA GPU acceleration and 100,000 character capacity",
            "tags_supported": False,
            "acceleration": "CUDA GPU (with CPU Fallback)",
            "is_downloaded": check_kokoro_installed(),
            "size": "350 MB",
            "hf_repo": "hexgrad/Kokoro-82M",
            "voices": [
                {"id": "af_heart", "name": "Heart (American Female)", "gender": "female", "avatar": "female"},
                {"id": "af_sarah", "name": "Sarah (American Female)", "gender": "female", "avatar": "female"},
                {"id": "af_bella", "name": "Bella (American Female)", "gender": "female", "avatar": "female"},
                {"id": "am_adam", "name": "Adam (American Male)", "gender": "male", "avatar": "male"},
                {"id": "am_michael", "name": "Michael (American Male)", "gender": "male", "avatar": "male"},
                {"id": "bf_emma", "name": "Emma (British Female)", "gender": "female", "avatar": "female"},
                {"id": "bm_george", "name": "George (British Male)", "gender": "male", "avatar": "male"},
            ],
        },
        {
            "id": "whisper-base",
            "name": "Whisper STT (Base)",
            "type": "stt",
            "engine": "whisper",
            "description": "High accuracy local speech-to-text transcription",
            "tags_supported": False,
            "acceleration": "CPU / CUDA",
            "is_downloaded": check_whisper_installed(),
            "size": "145 MB",
            "hf_repo": "Systran/faster-whisper-base",
            "voices": [],
        },
    ]

    return all_models


# Download Job State
@dataclass
class SpeechDownloadJob:
    job_id: str
    model_id: str
    status: str
    progress: float
    message: str
    error: Optional[str] = None


_download_jobs: Dict[str, SpeechDownloadJob] = {}
_jobs_lock = threading.Lock()


# Speech Synthesis Background Job State
@dataclass
class SpeechSynthesisJob:
    job_id: str
    model_id: str
    voice: str
    text_length: int
    total_chunks: int
    current_chunk: int
    progress: float
    status: str  # "queued", "processing", "completed", "failed", "cancelled"
    message: str
    error: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None


_speech_synthesis_jobs: Dict[str, SpeechSynthesisJob] = {}
_speech_jobs_lock = threading.Lock()


def get_speech_synthesis_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _speech_jobs_lock:
        job = _speech_synthesis_jobs.get(job_id)
        if not job:
            return None
        d = asdict(job)
        d.pop("audio_bytes", None)  # Do not serialize large WAV byte payload
        if job.status == "completed":
            d["audio_url"] = f"/v1/audio/speech/jobs/{job_id}/result"
        return d


def get_speech_synthesis_job_audio(job_id: str) -> Optional[bytes]:
    with _speech_jobs_lock:
        job = _speech_synthesis_jobs.get(job_id)
        if not job or job.status != "completed":
            return None
        return job.audio_bytes


def cancel_speech_synthesis_job(job_id: str) -> bool:
    with _speech_jobs_lock:
        job = _speech_synthesis_jobs.get(job_id)
        if job and job.status in ("queued", "processing"):
            job.status = "cancelled"
            job.message = "Synthesis cancelled by user."
            return True
        return False


def get_download_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _download_jobs.get(job_id)
        return asdict(job) if job else None


def start_model_download(model_id: str) -> str:
    job_id = str(uuid.uuid4())
    job = SpeechDownloadJob(
        job_id=job_id,
        model_id=model_id,
        status="running",
        progress=0.0,
        message="Starting download...",
    )
    with _jobs_lock:
        _download_jobs[job_id] = job

    def worker():
        try:
            if model_id == "kokoro-82m":
                dest = get_speech_models_dir() / "kokoro-82m"
                dest.mkdir(parents=True, exist_ok=True)
                with _jobs_lock:
                    job.message = "Downloading Kokoro ONNX and voice weights..."
                    job.progress = 0.3

                import urllib.request
                model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
                voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

                urllib.request.urlretrieve(model_url, dest / "kokoro-v1.0.onnx")
                with _jobs_lock:
                    job.progress = 0.7

                urllib.request.urlretrieve(voices_url, dest / "voices-v1.0.bin")
                with _jobs_lock:
                    job.progress = 1.0
                    job.status = "completed"
                    job.message = "Kokoro-82M download complete."

            elif model_id == "whisper-base":
                with _jobs_lock:
                    job.message = "Downloading Whisper Base weights..."
                    job.progress = 0.5
                from faster_whisper import WhisperModel
                _ = WhisperModel("base", device="cpu", compute_type="int8")
                with _jobs_lock:
                    job.progress = 1.0
                    job.status = "completed"
                    job.message = "Whisper Base model ready."

            else:
                raise ValueError(f"Unknown speech model {model_id}")

            invalidate_speech_models_cache()

        except Exception as e:
            logger.error("Speech model download failed: %s", e)
            with _jobs_lock:
                job.status = "failed"
                job.error = str(e)
                job.message = f"Download failed: {e}"

    threading.Thread(target=worker, daemon=True).start()
    return job_id


# =========================================================================
# TTS Synthesis Service
# =========================================================================

def _get_kokoro_engine():
    global _kokoro_engine
    if _kokoro_engine is not None:
        return _kokoro_engine

    models_dir = get_speech_models_dir() / "kokoro-82m"
    onnx_file = models_dir / "kokoro-v1.0.onnx"
    voices_file = models_dir / "voices-v1.0.bin"

    if not onnx_file.exists() or not voices_file.exists():
        legacy_dir = Path("D:/Ragul/oprel-SDK/.speech_models/kokoro-82m")
        if (legacy_dir / "kokoro-v1.0.onnx").exists() and (legacy_dir / "voices-v1.0.bin").exists():
            models_dir = legacy_dir
            onnx_file = legacy_dir / "kokoro-v1.0.onnx"
            voices_file = legacy_dir / "voices-v1.0.bin"
        else:
            raise FileNotFoundError(f"Kokoro-82M model files missing in {models_dir}")

    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    available_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # Suppress benign graph layout warnings
    sess = ort.InferenceSession(str(onnx_file), sess_options=sess_options, providers=providers)
    logger.info("Initialized Kokoro-82M engine with active providers: %s", sess.get_providers())

    _kokoro_engine = Kokoro.from_session(sess, str(voices_file))
    return _kokoro_engine


def _split_text_into_chunks(text: str, max_chunk_size: int = 1200) -> List[str]:
    """
    Intelligently splits long text up to 100,000 characters into natural chunks
    bounded by paragraphs or sentence terminals (. ! ?) to avoid phonemizer buffer overflows.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    paragraphs = text.split("\n")
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 1 <= max_chunk_size:
            current_chunk = (current_chunk + " " + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            if len(para) <= max_chunk_size:
                current_chunk = para
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current_chunk) + len(sent) + 1 <= max_chunk_size:
                        current_chunk = (current_chunk + " " + sent).strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = ""
                        if len(sent) <= max_chunk_size:
                            current_chunk = sent
                        else:
                            for i in range(0, len(sent), max_chunk_size):
                                chunks.append(sent[i : i + max_chunk_size])
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def synthesize_kokoro_audio(text: str, voice: str = "af_heart", speed: float = 1.0) -> bytes:
    """
    Synthesizes speech using Kokoro-82M on CUDA GPU (with CPU fallback).
    Supports large documents and texts up to 100,000 characters.
    Strips expressive bracket tags like [laughing] since Kokoro is a plain-text acoustic model.
    """
    if len(text) > 100000:
        raise ValueError(
            f"Input text exceeds the maximum allowed limit of 100,000 characters for Kokoro (received {len(text):,} chars)."
        )

    clean_text = re.sub(r"\[.*?\]", "", text).strip()
    if not clean_text:
        clean_text = text.replace("[", "").replace("]", "").strip()

    if not clean_text:
        raise ValueError("Cannot synthesize empty text.")

    chunks = _split_text_into_chunks(clean_text, max_chunk_size=1200)
    if not chunks:
        raise ValueError("Cannot synthesize empty text.")

    logger.info(
        "[Speech] Kokoro starting synthesis: %d characters across %d chunks (voice=%s, speed=%.1f)",
        len(clean_text),
        len(chunks),
        voice,
        speed,
    )

    with _engine_lock:
        kokoro = _get_kokoro_engine()
        sample_rate = 24000
        audio_segments: List[np.ndarray] = []

        for i, chunk in enumerate(chunks):
            chunk_t0 = time_module.time()
            samples, sr = kokoro.create(
                chunk,
                voice=voice or "af_heart",
                speed=speed,
                lang="en-us",
            )
            sample_rate = sr
            if samples is not None and len(samples) > 0:
                audio_segments.append(samples)

            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            elapsed = time_module.time() - chunk_t0
            logger.info(
                "[Speech] Synthesized chunk %d/%d (%d%%) in %.2fs",
                i + 1,
                len(chunks),
                int((i + 1) / len(chunks) * 100),
                elapsed,
            )

        if not audio_segments:
            raise RuntimeError("Kokoro audio synthesis produced no audio frames.")

        full_audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]

        buffer = io.BytesIO()
        sf.write(buffer, full_audio, sample_rate, format="WAV")
        logger.info(
            "[Speech] Finished Kokoro synthesis: %d chunks, total audio %.2fs",
            len(chunks),
            len(full_audio) / sample_rate,
        )
        return buffer.getvalue()


def start_speech_synthesis_job(
    model: str,
    text: str,
    voice: Optional[str] = None,
    speed: float = 1.0,
) -> str:
    """
    Submits a background speech synthesis job and returns job_id immediately.
    Ideal for long-text generation up to 100,000 characters with real-time progress.
    """
    job_id = str(uuid.uuid4())
    print(f"\n[OPREL] [Speech] >>> Job queued: id={job_id[:8]} | model={model} | chars={len(text)} | voice={voice or 'default'}")
    job = SpeechSynthesisJob(
        job_id=job_id,
        model_id=model,
        voice=voice or "af_heart",
        text_length=len(text),
        total_chunks=1,
        current_chunk=0,
        progress=0.0,
        status="queued",
        message="Queued for synthesis...",
        created_at=time_module.time(),
    )
    with _speech_jobs_lock:
        _speech_synthesis_jobs[job_id] = job

    thread = threading.Thread(
        target=_run_speech_synthesis_job,
        args=(job_id, model, text, voice, speed),
        daemon=True,
    )
    thread.start()
    return job_id


def _run_speech_synthesis_job(
    job_id: str,
    model: str,
    text: str,
    voice: Optional[str],
    speed: float,
):
    start_time = time_module.time()
    print(f"[OPREL] [Speech] >>> Worker thread started for [{job_id[:8]}]: model={model}, chars={len(text)}")
    logger.info("[Speech Job %s] Received synthesis task: model=%s, chars=%d", job_id, model, len(text))
    try:
        model_lower = model.lower()
        # Kokoro-82M multi-chunk job
        if len(text) > 100000:
            raise ValueError(
                f"Kokoro-82M input exceeds maximum supported limit of 100,000 characters (received {len(text):,})."
            )

        clean_text = re.sub(r"\[.*?\]", "", text).strip()
        if not clean_text:
            clean_text = text.replace("[", "").replace("]", "").strip()

        if not clean_text:
            raise ValueError("Cannot synthesize empty text.")

        chunks = _split_text_into_chunks(clean_text, max_chunk_size=1200)
        total_chunks = max(1, len(chunks))

        with _speech_jobs_lock:
            job = _speech_synthesis_jobs[job_id]
            job.status = "processing"
            job.total_chunks = total_chunks
            job.current_chunk = 0
            job.progress = 0.05
            job.message = f"Starting synthesis: 0 of {total_chunks} chunks (0%)..."

        logger.info("[Speech Job %s] Kokoro processing %d chunks for %d characters", job_id, total_chunks, len(clean_text))

        with _engine_lock:
            kokoro = _get_kokoro_engine()
            sample_rate = 24000
            audio_segments: List[np.ndarray] = []

            for i, chunk in enumerate(chunks):
                # Check for cancellation
                with _speech_jobs_lock:
                    if _speech_synthesis_jobs[job_id].status == "cancelled":
                        logger.info("[Speech Job %s] Aborted by user at chunk %d/%d", job_id, i + 1, total_chunks)
                        return

                chunk_t0 = time_module.time()
                samples, sr = kokoro.create(
                    chunk,
                    voice=voice or "af_heart",
                    speed=speed,
                    lang="en-us",
                )
                sample_rate = sr
                if samples is not None and len(samples) > 0:
                    audio_segments.append(samples)

                # Synchronize CUDA stream & flush intermediate allocations
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                current = i + 1
                prog = round(current / total_chunks, 2)
                pct = int(prog * 100)
                elapsed = time_module.time() - chunk_t0
                logger.info(
                    "[Speech Job %s] Chunk %d/%d completed in %.2fs (%d%%)",
                    job_id,
                    current,
                    total_chunks,
                    elapsed,
                    pct,
                )

                with _speech_jobs_lock:
                    job = _speech_synthesis_jobs[job_id]
                    job.current_chunk = current
                    job.progress = prog
                    job.message = f"Synthesizing chunk {current} of {total_chunks} ({pct}%)..."

            if not audio_segments:
                raise RuntimeError("Kokoro synthesis produced no audio frames.")

            full_audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
            buf = io.BytesIO()
            sf.write(buf, full_audio, sample_rate, format="WAV")
            audio_bytes = buf.getvalue()

            with _speech_jobs_lock:
                job = _speech_synthesis_jobs[job_id]
                job.status = "completed"
                job.progress = 1.0
                job.message = f"Synthesis completed ({len(audio_segments)} chunks in {round(time_module.time() - start_time, 1)}s)"
                job.audio_bytes = audio_bytes
                job.completed_at = time_module.time()

            logger.info(
                "[Speech Job %s] COMPLETED: %d chunks in %.2fs (audio duration: %.2fs)",
                job_id,
                total_chunks,
                time_module.time() - start_time,
                len(full_audio) / sample_rate,
            )

    except Exception as exc:
        logger.error("[Speech Job %s] Failed: %s", job_id, exc)
        with _speech_jobs_lock:
            job = _speech_synthesis_jobs[job_id]
            job.status = "failed"
            job.error = str(exc)
            job.message = f"Synthesis failed: {exc}"
            job.completed_at = time_module.time()


_whisper_engine = None
_whisper_loading = False
_whisper_lock = threading.Lock()


def is_whisper_ready() -> bool:
    """Returns True if the Whisper model is already loaded in memory."""
    return _whisper_engine is not None


def warmup_whisper_model() -> dict:
    """
    Pre-loads the Faster-Whisper model into GPU/CPU memory in a background thread.
    Safe to call multiple times (no-ops if already loaded or loading).
    Returns immediately with status; poll is_whisper_ready() or GET /v1/audio/whisper/status.
    """
    global _whisper_engine, _whisper_loading

    if _whisper_engine is not None:
        return {"status": "ready", "message": "Whisper model already loaded"}

    if _whisper_loading:
        return {"status": "loading", "message": "Whisper model is loading"}

    def _load():
        global _whisper_engine, _whisper_loading
        with _whisper_lock:
            if _whisper_engine is not None:
                return
            _whisper_loading = True
            try:
                from faster_whisper import WhisperModel
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if torch.cuda.is_available() else "int8"
                try:
                    model = WhisperModel("base", device=device, compute_type=compute_type)
                except Exception:
                    model = WhisperModel("base", device="cpu", compute_type="int8")
                _whisper_engine = model
                logger.info("[Whisper] Model loaded on %s (%s)", device, compute_type)
            except Exception as exc:
                logger.error("[Whisper] Failed to warm up model: %s", exc)
            finally:
                _whisper_loading = False

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    return {"status": "loading", "message": "Whisper model loading started"}


def transcribe_audio_whisper(audio_bytes: bytes, beam_size: int = 5) -> str:
    """
    Transcribes spoken audio into text using local Faster-Whisper.
    Supports CUDA GPU acceleration with automatic CPU fallback.
    Thread-safe and non-blocking for FastAPI event loop.
    """
    from faster_whisper import WhisperModel

    unique_id = uuid.uuid4().hex[:8]
    temp_dir = CONFIG.cache_dir / "stt_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_audio = temp_dir / f"stt_{unique_id}.bin"
    temp_audio.write_bytes(audio_bytes)

    try:
        global _whisper_engine
        with _whisper_lock:
            if _whisper_engine is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if torch.cuda.is_available() else "int8"
                try:
                    _whisper_engine = WhisperModel("base", device=device, compute_type=compute_type)
                except Exception:
                    _whisper_engine = WhisperModel("base", device="cpu", compute_type="int8")

            segments, info = _whisper_engine.transcribe(str(temp_audio), beam_size=beam_size)
            text = " ".join(s.text.strip() for s in segments)
            return text.strip()
    finally:
        if temp_audio.exists():
            try:
                temp_audio.unlink(missing_ok=True)
            except Exception:
                pass
