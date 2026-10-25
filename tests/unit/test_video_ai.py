"""Shared AI helpers: privacy gate, caption chunking and narration fitting (fake TTS; validation uses the real Node validator)."""

from __future__ import annotations

import io
import wave

import pytest

from oprel.video import ai, projects, renderer_jobs

needs_renderer = pytest.mark.skipif(not renderer_jobs.renderer_status()["available"], reason="Node renderer deps not installed")


@pytest.fixture(autouse=True)
def isolated_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("OPREL_VIDEO_PROJECTS_DIR", str(tmp_path / "video_projects"))


def _noop(*_a):
    pass


def storyboard():
    return {
        "schemaVersion": 1, "title": "T",
        "scenes": [
            {"id": "hook", "type": "hook", "durationSeconds": 4, "narration": "Why do models forget? Because training stops.",
             "params": {"headline": "Why models forget"}},
            {"id": "end", "type": "conclusion", "durationSeconds": 4, "narration": "Retrieval adds fresh facts.",
             "params": {"title": "Retrieval helps", "points": ["Search", "Answer"]}},
        ],
    }


def test_caption_chunks_are_short_and_cover_the_text():
    text = "First sentence here. " + "word " * 60 + "Last one!"
    chunks = ai._split_caption_chunks(text)
    assert all(len(c) <= 95 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_cloud_models_need_explicit_consent(monkeypatch):
    monkeypatch.setattr(ai, "is_cloud_model", lambda m: m.startswith("openai::"))
    with pytest.raises(ai.AIStageError) as e:
        ai.check_model_request("openai::gpt", allow_cloud=False)
    assert e.value.code == "cloud_not_allowed"
    ai.check_model_request("openai::gpt", allow_cloud=True)
    ai.check_model_request("qwen:7b", allow_cloud=False)
    with pytest.raises(ai.AIStageError):
        ai.check_model_request("", allow_cloud=True)


def _fake_wav(seconds: float, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"\x01\x00" * int(sr * seconds))
    return buf.getvalue()


@needs_renderer
def test_narration_fits_scene_timing_and_builds_captions(monkeypatch):
    pid = projects.create_project("narr test", storyboard())["id"]
    monkeypatch.setattr(ai, "tts_models_ready", lambda: [{"id": "kokoro-82m"}])
    clips = iter([_fake_wav(5.0), _fake_wav(3.0)])
    monkeypatch.setattr(ai, "_synthesize", lambda *a, **k: next(clips))

    out = ai.narrate_project(pid, "kokoro-82m", None, 1.0, _noop, lambda: False)
    sb = projects.read_storyboard(pid)
    fps = 30
    # scene length = lead-in + real audio + lead-out, rounded up to whole frames
    assert sb["scenes"][0]["durationFrames"] == int(-(-(0.3 + 5.0 + 0.7) * fps // 1))
    assert sb["scenes"][1]["durationFrames"] == int(-(-(0.3 + 3.0 + 0.7) * fps // 1))
    assert "durationSeconds" not in sb["scenes"][0]
    total_frames = sum(s["durationFrames"] for s in sb["scenes"])
    wav = projects.project_dir(pid) / "assets" / "narration.wav"
    with wave.open(str(wav), "rb") as w:
        assert w.getnframes() == round(total_frames / fps * 24000)  # audio length == timeline length
    assert sb["captionsSource"] == "estimated-from-audio-duration"
    assert sb["captions"][0]["startMs"] == 300
    assert all(a["endMs"] <= b["startMs"] for a, b in zip(sb["captions"], sb["captions"][1:]))
    assert out["captions"] == len(sb["captions"])
    assert renderer_jobs.run_blocking("validate", pid)["event"] == "result"


def test_narration_refuses_when_tts_model_missing(monkeypatch):
    pid = projects.create_project("no tts", storyboard())["id"]
    monkeypatch.setattr(ai, "tts_models_ready", lambda: [])
    with pytest.raises(ai.AIStageError) as e:
        ai.narrate_project(pid, "kokoro-82m", None, 1.0, _noop, lambda: False)
    assert e.value.code == "tts_unavailable"


def test_narration_stops_when_cancelled(monkeypatch):
    pid = projects.create_project("cancel narr", storyboard())["id"]
    monkeypatch.setattr(ai, "tts_models_ready", lambda: [{"id": "kokoro-82m"}])
    monkeypatch.setattr(ai, "_synthesize", lambda *a, **k: _fake_wav(1.0))
    with pytest.raises(ai.AIStageError) as e:
        ai.narrate_project(pid, "kokoro-82m", None, 1.0, _noop, lambda: True)
    assert e.value.code == "cancelled"
