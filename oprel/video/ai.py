"""Shared helpers for the Video Studio agent: model calls, Audio Studio TTS, narration fitting, storyboard validation.

Everything here reuses existing Oprel services (``generate_text`` and the speech synthesizers).
"""

from __future__ import annotations

import io
import json
import math
import re
import wave
from typing import Any, Callable, Optional

from oprel.video import projects, renderer_jobs

MIN_SCENE_SECONDS = 2.0
LEAD_IN_S, LEAD_OUT_S = 0.3, 0.7


class AIStageError(RuntimeError):
    def __init__(self, code: str, message: str, issues: Optional[list] = None):
        super().__init__(message)
        self.code, self.issues = code, issues or []


def is_cloud_model(model_id: str) -> bool:
    """True when the model id resolves to a configured external provider (privacy gate)."""
    from oprel.server import db

    p_id = model_id.split("::", 1)[0] if "::" in model_id else model_id.split(":", 1)[0]
    return db.get_provider(p_id) is not None


def _format_issues(issues: list[dict]) -> str:
    return "\n".join(f"- {i.get('path') or 'storyboard'}: {i.get('message')}" for i in issues if i.get("level") == "error")


def _validate_candidate(project_id: str, candidate: dict[str, Any]) -> list[dict]:
    d = projects.project_dir(project_id)
    (d / "tmp").mkdir(exist_ok=True)
    (d / "tmp" / "candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
    res = renderer_jobs.run_blocking("validate", project_id, ["--file", "tmp/candidate.json"])
    if res.get("event") == "result":
        return []
    if res.get("code") == "invalid_project":
        return [i for i in res.get("issues", []) if i.get("level") == "error"]
    raise AIStageError("validator_failed", res.get("message") or "storyboard validator failed to run")


async def _ask_model(model_id: str, system: str, user: str, max_tokens: int, temperature: float = 0.4) -> str:
    from oprel.server.services.generation import GenerateParams, GenerateResult, generate_text

    result = await generate_text(
        GenerateParams(
            model_id=model_id, prompt=user, max_tokens=max_tokens, temperature=temperature, top_p=0.9, top_k=40,
            repeat_penalty=1.05, stream=False, images=None, conversation_id=None, system_prompt=system,
            reset_conversation=True, thinking=False, rag=False, ephemeral=True,
        )
    )
    if not isinstance(result, GenerateResult):
        raise AIStageError("bad_reply", "model returned a stream where text was expected")
    return result.text


def check_model_request(model_id: str, allow_cloud: bool) -> None:
    """Privacy gate: a configured cloud provider is only used with explicit consent."""
    if not (model_id or '').strip():
        raise AIStageError("bad_request", "choose a model")
    if is_cloud_model(model_id) and not allow_cloud:
        raise AIStageError("cloud_not_allowed", "this model is a cloud provider; the brief would be sent off-device. Confirm to allow it.")


# ── Narration ────────────────────────────────────────────────────────────────

def _split_caption_chunks(text: str, max_chars: int = 90) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    out: list[str] = []
    for s in sentences:
        words, cur = s.split(), ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                out.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            out.append(cur)
    return out


def _wav_info(data: bytes) -> tuple[int, int, int, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()


def tts_models_ready() -> list[dict[str, Any]]:
    from oprel.server.services.speech import get_speech_models_status

    return [m for m in get_speech_models_status() if m.get("type") == "tts" and m.get("is_downloaded")]


def _synthesize(model: str, text: str, voice: Optional[str], speed: float) -> bytes:
    from oprel.server.services.speech import synthesize_fish_audio, synthesize_kokoro_audio

    m = model.lower()
    if "kokoro" in m:
        return synthesize_kokoro_audio(text, voice=voice or "af_heart", speed=speed)
    if "fish" in m or "s1" in m:
        return synthesize_fish_audio(text)
    raise AIStageError("bad_request", f"unsupported narration model '{model}'")


def narrate_project(
    project_id: str, tts_model: str, voice: Optional[str], speed: float,
    progress: Callable[[str, float, str], None], cancelled: Callable[[], bool],
) -> dict[str, Any]:
    """Synthesize each scene's narration, fit scene durations to the real audio, write assets/narration.wav and captions.

    Scene boundaries are exact (measured from the audio). Caption sentence times inside a scene are *estimated*
    proportionally to text length because no forced aligner is available; the storyboard records this.
    """
    ready = {m["id"] for m in tts_models_ready()}
    if not any(tts_model.lower() in r or r in tts_model.lower() for r in ready):
        raise AIStageError("tts_unavailable", "that narration model is not installed. Download it on the Audio page first.")

    sb = projects.read_storyboard(project_id)
    scenes = sb["scenes"]
    spoken = [i for i, s in enumerate(scenes) if (s.get("narration") or "").strip()]
    if not spoken:
        raise AIStageError("no_narration", "no scene has narration text yet")
    fps = sb.get("format", {}).get("fps", 30)

    clips: dict[int, bytes] = {}
    fmt: Optional[tuple[int, int, int]] = None
    for n, i in enumerate(spoken):
        if cancelled():
            raise AIStageError("cancelled", "narration cancelled")
        progress("tts", 0.05 + 0.8 * n / len(spoken), f"Speaking scene {n + 1} of {len(spoken)}")
        wav = _synthesize(tts_model, scenes[i]["narration"].strip(), voice, speed)
        ch, sw, sr, nf = _wav_info(wav)
        if nf == 0:
            raise AIStageError("tts_empty", f"narration model returned no audio for scene '{scenes[i].get('id')}'")
        if fmt is None:
            fmt = (ch, sw, sr)
        elif fmt != (ch, sw, sr):
            raise AIStageError("tts_format", "narration clips have mismatched audio formats")
        clips[i] = wav

    assert fmt is not None
    ch, sw, sr = fmt
    progress("mix", 0.9, "Fitting scenes to narration and mixing audio")
    frame_bytes = ch * sw
    pcm = bytearray()
    captions: list[dict[str, Any]] = []
    start_frame = 0
    for i, scene in enumerate(scenes):
        if i in clips:
            with wave.open(io.BytesIO(clips[i]), "rb") as w:
                audio = w.readframes(w.getnframes())
                audio_s = w.getnframes() / sr
            dur_s = max(MIN_SCENE_SECONDS, LEAD_IN_S + audio_s + LEAD_OUT_S)
            dur_f = math.ceil(dur_s * fps)
            scene_start_sample = round(start_frame / fps * sr)
            lead = round(LEAD_IN_S * sr)
            # pad the narration track with silence up to this scene's start, then the clip
            pcm.extend(b"\x00" * max(0, (scene_start_sample + lead) * frame_bytes - len(pcm)))
            pcm.extend(audio)
            chunks = _split_caption_chunks(scene["narration"])
            total = sum(len(c) for c in chunks) or 1
            t0 = start_frame / fps + LEAD_IN_S
            acc = 0
            for c in chunks:
                a = t0 + audio_s * acc / total
                acc += len(c)
                b = t0 + audio_s * acc / total
                captions.append({"startMs": round(a * 1000), "endMs": max(round(b * 1000), round(a * 1000) + 1), "text": c, "granularity": "sentence"})
        else:
            dur_f = scene.get("durationFrames") or max(1, round(float(scene.get("durationSeconds", MIN_SCENE_SECONDS)) * fps))
        scene.pop("durationSeconds", None)
        scene["durationFrames"] = int(dur_f)
        start_frame += int(dur_f)

    total_samples = round(start_frame / fps * sr)
    pcm.extend(b"\x00" * max(0, total_samples * frame_bytes - len(pcm)))
    d = projects.project_dir(project_id)
    out = d / "assets" / "narration.wav"
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(bytes(pcm))
    tmp.replace(out)

    # captions must not overlap by even 1ms after rounding
    for a, b in zip(captions, captions[1:]):
        if b["startMs"] < a["endMs"]:
            a["endMs"] = max(a["startMs"] + 1, b["startMs"])
    sb["scenes"] = scenes
    sb["narration"] = {"file": "assets/narration.wav", "durationMs": round(len(pcm) / frame_bytes / sr * 1000), "volume": 1}
    sb["captions"] = captions
    sb["captionsSource"] = "estimated-from-audio-duration"

    issues = _validate_candidate(project_id, sb)
    if issues:
        raise AIStageError("invalid_storyboard", "storyboard after narration did not validate", issues)
    projects.write_storyboard(project_id, sb)
    progress("done", 1.0, "Narration ready")
    return {"scenes": len(scenes), "seconds": round(start_frame / fps, 1), "captions": len(captions)}


# ── context window ───────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Deliberately pessimistic token estimate (code and non-English text tokenize worse than ~4 chars/token)."""
    return int(len(text) / 3.0) + 8


async def model_context_tokens(model_id: str) -> Optional[int]:
    """The context window (tokens) a local model is actually running with, or None when unknown (cloud providers, errors).

    Loads the model if needed (generation would do that anyway) and asks the llama.cpp server for its ``n_ctx``.
    """
    import asyncio
    import urllib.request

    try:
        if is_cloud_model(model_id):
            return None
        from oprel.server.domain.state import get_state
        from oprel.server.services.generation import _resolve_model_id
        from oprel.server.services.models import load_model

        rid = _resolve_model_id(model_id)
        state = get_state()
        if rid not in state.models:
            await asyncio.to_thread(load_model, rid)
        port = getattr(getattr(state.models.get(rid), "_process", None), "port", None)
        if not port:
            return None

        def fetch() -> Optional[int]:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=5) as r:
                props = json.loads(r.read())
            n = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
            return int(n) if n else None

        return await asyncio.to_thread(fetch)
    except Exception:
        return None


_CTX_ERROR = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)", re.I)


def parse_context_error(message: str) -> Optional[tuple[int, int]]:
    """(requested_tokens, available_tokens) from a llama.cpp "exceeds the available context size" error, else None."""
    m = _CTX_ERROR.search(message or "")
    return (int(m.group(1)), int(m.group(2))) if m else None
