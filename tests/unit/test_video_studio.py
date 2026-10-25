"""Video Studio: project storage, job lifecycle and (when Node deps are installed) a real render."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from oprel.video import projects, renderer_jobs

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "video" / "smoke_storyboard.json"


@pytest.fixture(autouse=True)
def isolated_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("OPREL_VIDEO_PROJECTS_DIR", str(tmp_path / "video_projects"))


@pytest.mark.parametrize("bad", ["../x", "a/b", "A_B", "..", "", "ab", "x" * 80, "C:\\x", "a b c"])
def test_project_id_rejects_traversal_and_junk(bad):
    with pytest.raises(projects.ProjectError):
        projects.project_dir(bad)


def test_create_read_update_delete_roundtrip():
    meta = projects.create_project("How RAG works", json.loads(SAMPLE.read_text()))
    pid = meta["id"]
    d = projects.project_dir(pid)
    assert all((d / s).is_dir() for s in projects.SUBDIRS)
    assert projects.read_storyboard(pid)["title"] == "Oprel Video Studio smoke test"

    sb = projects.read_storyboard(pid)
    sb["scenes"][0]["params"]["headline"] = "Edited"
    projects.write_storyboard(pid, sb)
    assert projects.read_storyboard(pid)["scenes"][0]["params"]["headline"] == "Edited"
    assert (d / "storyboard.prev.json").is_file()  # previous version kept
    assert [m["id"] for m in projects.list_projects()] == [pid]

    projects.delete_project(pid)
    assert not d.exists()
    with pytest.raises(projects.ProjectError):
        projects.get_project(pid)


def test_new_project_starts_empty_with_no_default_storyboard():
    pid = projects.create_project("blank")["id"]
    assert projects.read_storyboard(pid) is None
    assert not (projects.project_dir(pid) / "storyboard.json").exists()


def test_create_project_rejects_empty_title():
    with pytest.raises(projects.ProjectError):
        projects.create_project("   ")


def test_start_render_rejects_bad_quality_and_unknown_project():
    with pytest.raises(ValueError):
        renderer_jobs.start_render("some-project", quality="ultra")
    with pytest.raises(projects.ProjectError):
        renderer_jobs.start_render("missing-project", quality="preview")


def test_job_public_is_json_serialisable_while_process_attached():
    import subprocess, sys

    job = renderer_jobs.RenderJob(id="j1", project_id="p-1-abc", op="render", quality="preview")
    job._proc = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        out = json.dumps(job.public())
    finally:
        job._proc.wait()
    assert '"status": "queued"' in out and "_proc" not in out


def test_worker_unavailable_is_reported(monkeypatch):
    monkeypatch.setattr(renderer_jobs, "renderer_status", lambda: {"available": False, "message": "no node"})
    meta = projects.create_project("x video")
    with pytest.raises(renderer_jobs.RendererUnavailable):
        renderer_jobs.start_render(meta["id"], "preview")


needs_renderer = pytest.mark.skipif(not renderer_jobs.renderer_status()["available"], reason="Node renderer deps not installed")


def _wait(job_id, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        job = renderer_jobs.get_job(job_id)
        if job.status in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise AssertionError("render timed out")


@needs_renderer
def test_validate_reports_errors_for_bad_storyboard():
    meta = projects.create_project("bad one", json.loads(SAMPLE.read_text()))
    sb = projects.read_storyboard(meta["id"])
    sb["scenes"].append(dict(sb["scenes"][0]))  # duplicate id
    projects.write_storyboard(meta["id"], sb)
    res = renderer_jobs.run_blocking("validate", meta["id"])
    assert res["event"] == "error" and res["code"] == "invalid_project" and res["exitCode"] != 0
    assert any("duplicate" in i["message"] for i in res["issues"])


@needs_renderer
@pytest.mark.slow
def test_real_render_produces_validated_mp4_and_refuses_overwrite():
    meta = projects.create_project("render test", json.loads(SAMPLE.read_text()))
    pid = meta["id"]
    job = _wait(renderer_jobs.start_render(pid, "preview", frames="0-29").id)
    assert job.status == "completed", job.error
    out = projects.project_dir(pid) / "output" / "preview.mp4"
    assert out.is_file()
    probe = job.result["probe"]["video"]
    assert (probe["width"], probe["height"], probe["frames"], probe["codec"]) == (540, 960, 30, "h264")
    report = json.loads((projects.project_dir(pid) / "output" / "preview.validation.json").read_text())
    assert report["ok"] and all(c["ok"] for c in report["checks"])

    again = _wait(renderer_jobs.start_render(pid, "preview", frames="0-29").id)
    assert again.status == "failed" and again.error["code"] == "output_exists"
    assert out.is_file()  # previous output untouched


@needs_renderer
@pytest.mark.slow
def test_cancel_stops_render_and_leaves_no_output():
    meta = projects.create_project("cancel test", json.loads(SAMPLE.read_text()))
    pid = meta["id"]
    job = renderer_jobs.start_render(pid, "production")
    time.sleep(4)
    assert renderer_jobs.cancel_job(job.id)
    job = _wait(job.id, timeout=60)
    assert job.status == "cancelled"
    assert not (projects.project_dir(pid) / "output" / "final.mp4").exists()
