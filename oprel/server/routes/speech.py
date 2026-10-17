from __future__ import annotations

import asyncio
import io
import uuid
import wave
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from oprel.server.services.context import logger
from oprel.server.db import (
    AUDIO_HISTORY_DIR,
    save_audio_history,
    get_audio_history,
    get_audio_history_item,
    delete_audio_history_item,
    clear_audio_history,
)
from oprel.server.services.speech import (
    AVAILABLE_TAGS,
    cancel_speech_synthesis_job,
    get_download_job,
    get_speech_models_status,
    get_speech_synthesis_job,
    get_speech_synthesis_job_audio,
    is_whisper_ready,
    warmup_whisper_model,
    start_model_download,
    start_speech_synthesis_job,
    synthesize_fish_audio,
    synthesize_kokoro_audio,
    transcribe_audio_whisper,
)

router = APIRouter(tags=["Audio"])


class SpeechRequest(BaseModel):
    model: str = Field(default="fish-s1-mini", description="Speech model ID ('fish-s1-mini' or 'kokoro-82m')")
    input: str = Field(..., description="Text content to speak")
    voice: Optional[str] = Field(default=None, description="Voice identifier")
    speed: Optional[float] = Field(default=1.0, description="Speech rate multiplier")
    response_format: Optional[str] = Field(default="wav", description="Audio format (wav)")


class DownloadModelRequest(BaseModel):
    model: str = Field(..., description="Model ID to download ('fish-s1-mini', 'kokoro-82m', 'whisper-base')")


def _save_and_record_audio(
    audio_bytes: bytes,
    model: str,
    voice: Optional[str],
    text: str,
    speed: float = 1.0,
) -> dict:
    """Helper to save generated audio to disk and record it in the database."""
    history_id = str(uuid.uuid4())
    file_path = AUDIO_HISTORY_DIR / f"{history_id}.wav"
    try:
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
    except Exception as exc:
        logger.warning("Failed to save audio file to disk: %s", exc)

    duration_seconds = 0.0
    try:
        with io.BytesIO(audio_bytes) as b:
            with wave.open(b, "rb") as wf:
                duration_seconds = round(wf.getnframes() / float(wf.getframerate()), 2)
    except Exception:
        pass

    model_lower = (model or "").lower()
    model_name = "Kokoro-82M" if "kokoro" in model_lower else "Fish Audio S1-mini"
    voice_id = voice or ("af_heart" if "kokoro" in model_lower else "default")
    voice_name = voice_id.replace("_", " ").title()

    return save_audio_history(
        id=history_id,
        model_id=model,
        model_name=model_name,
        voice_id=voice_id,
        voice_name=voice_name,
        text=text,
        audio_path=str(file_path),
        duration_seconds=duration_seconds,
        speed=speed,
    )


@router.get("/v1/audio/models")
async def list_speech_models():
    """
    Returns available speech models for the Audio Studio.
    These models do NOT appear in the generic Model Registry page.
    """
    return {"data": get_speech_models_status()}


@router.get("/v1/audio/tags")
async def list_speech_tags(model: Optional[str] = None):
    """
    Returns the supported expressive tags for Fish Audio S1-mini.
    """
    return {"tags": AVAILABLE_TAGS}


@router.post("/v1/audio/models/download")
async def trigger_speech_model_download(req: DownloadModelRequest):
    """
    Triggers a background download for an audio model directly to the speech directory.
    """
    try:
        job_id = start_model_download(req.model)
        return {"job_id": job_id, "status": "running", "message": f"Download initiated for {req.model}"}
    except Exception as exc:
        logger.error("Failed to initiate audio model download: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/v1/audio/models/jobs/{job_id}")
async def get_speech_model_job_status(job_id: str):
    """
    Checks the progress of an active speech model download job.
    """
    job = get_download_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    """
    OpenAI-compatible text-to-speech synthesis endpoint.
    Routes to Fish Audio S1-mini (CUDA GPU, max 3,000 chars)
    or Kokoro-82M (CUDA GPU accelerated up to 100,000 chars).
    """
    print(f"\n[OPREL] [Speech] >>> Received direct synthesis request: model={req.model}, chars={len(req.input)}, voice={req.voice}")
    try:
        model_id = (req.model or "fish-s1-mini").lower()

        if "fish" in model_id or "s1" in model_id:
            if len(req.input) > 3000:
                raise HTTPException(
                    status_code=400,
                    detail=f"Fish Audio S1-mini input exceeds the safe limit of 3,000 characters (received {len(req.input):,}). Please use Kokoro-82M for large texts up to 100,000 characters.",
                )
            audio_bytes = synthesize_fish_audio(req.input)
        elif "kokoro" in model_id:
            if len(req.input) > 100000:
                raise HTTPException(
                    status_code=400,
                    detail=f"Kokoro-82M input exceeds the maximum supported limit of 100,000 characters (received {len(req.input):,}).",
                )
            audio_bytes = synthesize_kokoro_audio(req.input, voice=req.voice or "af_heart", speed=req.speed or 1.0)
        else:
            # Fallback to fish audio if installed, otherwise kokoro
            try:
                if len(req.input) > 3000:
                    audio_bytes = synthesize_kokoro_audio(req.input, voice=req.voice or "af_heart", speed=req.speed or 1.0)
                else:
                    audio_bytes = synthesize_fish_audio(req.input)
            except Exception:
                audio_bytes = synthesize_kokoro_audio(req.input, voice=req.voice or "af_heart", speed=req.speed or 1.0)

        # Save to database and persistent disk
        rec = _save_and_record_audio(
            audio_bytes=audio_bytes,
            model=req.model or "fish-s1-mini",
            voice=req.voice,
            text=req.input,
            speed=req.speed or 1.0,
        )

        headers = {
            "X-Audio-History-Id": rec.get("id", ""),
            "X-Audio-Duration": str(rec.get("duration_seconds", 0.0)),
        }
        return Response(content=audio_bytes, media_type="audio/wav", headers=headers)

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.error("Speech synthesis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/v1/audio/whisper/warmup")
async def whisper_warmup():
    """
    Pre-loads the Faster-Whisper model in a background thread.
    The frontend calls this immediately when the STT tab opens so that by the
    time the user clicks 'Finish', the model is already warm and the first
    transcription completes in ~1-2 seconds instead of 60-120 seconds.
    """
    result = warmup_whisper_model()
    return result


@router.get("/v1/audio/whisper/status")
async def whisper_status():
    """
    Returns whether the Whisper model is ready to transcribe.
    Frontend polls this (1.5s interval) after calling /warmup to
    transition from 'Connecting...' to 'Click to start transcribing'.
    """
    ready = is_whisper_ready()
    return {"ready": ready, "status": "ready" if ready else "loading"}


@router.post("/v1/audio/speech/jobs")
async def create_speech_job_endpoint(req: SpeechRequest):
    """
    Starts an asynchronous background speech synthesis job.
    Enables long-form text synthesis up to 100,000 characters with live chunk-by-chunk progress tracking.
    """
    print(f"\n[OPREL] [Speech] >>> Received speech job request: model={req.model}, chars={len(req.input)}, voice={req.voice}")
    model_id = (req.model or "kokoro-82m").lower()
    if "fish" in model_id or "s1" in model_id:
        if len(req.input) > 3000:
            raise HTTPException(
                status_code=400,
                detail=f"Fish Audio S1-mini input exceeds the safe limit of 3,000 characters (received {len(req.input):,}). Please use Kokoro-82M for large texts up to 100,000 characters.",
            )
    elif "kokoro" in model_id:
        if len(req.input) > 100000:
            raise HTTPException(
                status_code=400,
                detail=f"Kokoro-82M input exceeds the maximum supported limit of 100,000 characters (received {len(req.input):,}).",
            )

    job_id = start_speech_synthesis_job(
        model=req.model,
        text=req.input,
        voice=req.voice,
        speed=req.speed or 1.0,
    )
    return {"job_id": job_id, "status": "queued"}


@router.get("/v1/audio/speech/jobs/{job_id}")
async def get_speech_job_endpoint(job_id: str):
    """
    Polls the real-time status and chunk progress of an active speech synthesis job.
    """
    job = get_speech_synthesis_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Speech job not found")
    return job


@router.get("/v1/audio/speech/jobs/{job_id}/result")
async def get_speech_job_result_endpoint(job_id: str):
    """
    Returns the completed WAV audio byte stream for a finished speech job and records to history.
    """
    audio_bytes = get_speech_synthesis_job_audio(job_id)
    if not audio_bytes:
        job = get_speech_synthesis_job(job_id)
        if job and job.get("status") == "processing":
            raise HTTPException(status_code=202, detail="Speech synthesis still in progress")
        if job and job.get("status") == "failed":
            raise HTTPException(status_code=500, detail=job.get("error", "Job failed"))
        raise HTTPException(status_code=404, detail="Audio result not found or expired")

    job = get_speech_synthesis_job(job_id)
    headers = {}
    if job:
        rec = _save_and_record_audio(
            audio_bytes=audio_bytes,
            model=job.get("model_id", "kokoro-82m"),
            voice=job.get("voice"),
            text=job.get("text", ""),
            speed=job.get("speed", 1.0),
        )
        headers = {
            "X-Audio-History-Id": rec.get("id", ""),
            "X-Audio-Duration": str(rec.get("duration_seconds", 0.0)),
        }

    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


@router.post("/v1/audio/speech/jobs/{job_id}/cancel")
async def cancel_speech_job_endpoint(job_id: str):
    """
    Cancels an active speech synthesis job.
    """
    cancelled = cancel_speech_synthesis_job(job_id)
    return {"job_id": job_id, "cancelled": cancelled}


@router.get("/v1/audio/history")
async def list_audio_history(days: Optional[int] = None):
    """
    Returns audio generation history, optionally filtered by days (7, 30, 90).
    """
    items = get_audio_history(days=days)
    return {"data": items}


@router.get("/v1/audio/history/{id}/audio")
async def get_audio_history_audio(id: str):
    """
    Streams the stored WAV audio file for an audio generation record.
    """
    item = get_audio_history_item(id)
    if not item or not item.get("audio_path"):
        raise HTTPException(status_code=404, detail="Audio file record not found")
    path = Path(item["audio_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing on disk")
    return FileResponse(path, media_type="audio/wav", filename=f"speech_{id}.wav")


@router.delete("/v1/audio/history/{id}")
async def delete_audio_history_route(id: str):
    """
    Deletes an audio generation record and its audio file.
    """
    success = delete_audio_history_item(id)
    if not success:
        raise HTTPException(status_code=404, detail="Audio record not found")
    return {"success": True, "id": id}


@router.delete("/v1/audio/history")
async def clear_audio_history_route():
    """
    Clears all audio generation history and audio files.
    """
    clear_audio_history()
    return {"success": True}


@router.post("/v1/audio/transcriptions")
async def create_transcription(file: UploadFile = File(...), fast: Optional[bool] = False):
    """
    OpenAI-compatible speech-to-text transcription endpoint using local Whisper.
    Supports fast=True for responsive real-time streaming chunks (beam_size=1).
    """
    try:
        audio_content = await file.read()
        beam_size = 1 if fast else 5
        transcribed_text = await asyncio.to_thread(transcribe_audio_whisper, audio_content, beam_size=beam_size)
        return {"text": transcribed_text}
    except Exception as exc:
        logger.error("Speech transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

