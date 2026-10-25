"""Render job orchestration.

Rendering runs in a separate Node child process (``renderer/bin/worker.ts``) so it never blocks the
FastAPI event loop, the model runtime or Audio Studio. This module only starts that process, parses its
NDJSON events and exposes typed job state. Arguments are always passed as a list (no shell).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from oprel.video import projects

RENDERER_DIR = Path(__file__).resolve().parent / "renderer"
WORKER = RENDERER_DIR / "bin" / "worker.ts"

QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED = "queued", "running", "completed", "failed", "cancelled"
_TERMINAL = {COMPLETED, FAILED, CANCELLED}

# One render at a time by default; each render already uses several browser tabs.
_MAX_CONCURRENT = max(1, int(os.environ.get("OPREL_VIDEO_MAX_JOBS", "1")))
_slots = threading.BoundedSemaphore(_MAX_CONCURRENT)


class RendererUnavailable(RuntimeError):
    pass


@dataclass
class RenderJob:
    id: str
    project_id: str
    op: str
    quality: str
    status: str = QUEUED
    stage: str = "queued"
    progress: float = 0.0
    message: str = ""
    error: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    _task: Any = field(default=None, repr=False)  # asyncio.Task for AI stages
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        # Not dataclasses.asdict(): it would try to deep-copy the Popen handle and the Event.
        return {
            "id": self.id, "project_id": self.project_id, "op": self.op, "quality": self.quality,
            "status": self.status, "stage": self.stage, "progress": self.progress, "message": self.message,
            "error": self.error, "result": self.result, "created_at": self.created_at, "finished_at": self.finished_at,
        }


_jobs: dict[str, RenderJob] = {}
_lock = threading.Lock()


def renderer_status() -> dict[str, Any]:
    """Report whether the worker can run, with an actionable message when it cannot."""
    node = shutil.which("node")
    deps = (RENDERER_DIR / "node_modules" / "remotion").is_dir()
    ok = bool(node) and deps and WORKER.is_file()
    msg = None
    if not node:
        msg = "Node.js was not found on PATH (Node 20+ required for Video Studio rendering)."
    elif not deps:
        msg = f"Renderer dependencies are not installed. Run `npm install` in {RENDERER_DIR}."
    return {"available": ok, "node": node, "rendererDir": str(RENDERER_DIR), "message": msg}


def _worker_cmd(op: str, project_dir: Path, extra: list[str]) -> list[str]:
    st = renderer_status()
    if not st["available"]:
        raise RendererUnavailable(st["message"] or "renderer unavailable")
    return [st["node"], "--import", "tsx", str(WORKER), op, "--project", str(project_dir), *extra]


def run_blocking(op: str, project_id: str, extra: Optional[list[str]] = None, timeout: float = 120) -> dict[str, Any]:
    """Run a short worker op (validate / compile / inspect) and return its final event."""
    d = projects.project_dir(project_id)
    cmd = _worker_cmd(op, d, extra or [])
    proc = subprocess.run(cmd, cwd=RENDERER_DIR, capture_output=True, text=True, timeout=timeout)
    last: dict[str, Any] = {"event": "error", "code": "no_output", "message": proc.stderr.strip()[-500:]}
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") in ("result", "error"):
            last = ev
    last["exitCode"] = proc.returncode
    return last


def run_worker(
    op: str, project_id: str, extra: Optional[list[str]] = None, timeout: float = 900,
    cancel: Optional[threading.Event] = None, holder: Optional[list] = None,
) -> dict[str, Any]:
    """Run a worker op to completion, streaming its events. Kills the process on cancel/timeout.

    ``holder`` (optional list) receives the Popen so another thread can terminate it.
    Returns the final ``result``/``error`` event with ``exitCode`` set.
    """
    d = projects.project_dir(project_id)
    cmd = _worker_cmd(op, d, extra or [])
    proc = subprocess.Popen(
        cmd, cwd=RENDERER_DIR, env={**os.environ, "NODE_NO_WARNINGS": "1"}, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    if holder is not None:
        holder.append(proc)
    final: dict[str, Any] = {"event": "error", "code": "no_output", "message": "worker produced no result"}
    tail: list[str] = []

    def pump() -> None:
        nonlocal final
        assert proc.stdout
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                tail.append(line.rstrip()[:300])
                del tail[:-8]
                continue
            if ev.get("event") in ("result", "error"):
                final = ev

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while t.is_alive():
        t.join(0.5)
        if cancel is not None and cancel.is_set():
            proc.kill()
            t.join(5)
            final = {"event": "error", "code": "cancelled", "message": "cancelled"}
            break
        if time.time() > deadline:
            proc.kill()
            t.join(5)
            final = {"event": "error", "code": "timeout", "message": f"worker exceeded {int(timeout)}s"}
            break
    code = proc.wait()
    if final.get("code") == "no_output" and tail:
        final["message"] = " | ".join(tail)[-400:]
    final["exitCode"] = code
    return final


def start_render(project_id: str, quality: str = "production", frames: Optional[str] = None, overwrite: bool = False) -> RenderJob:
    if quality not in ("preview", "production"):
        raise ValueError("quality must be 'preview' or 'production'")
    d = projects.project_dir(project_id)  # validates the id
    if not (d / "project.json").is_file():
        raise projects.ProjectError("project not found")
    extra = ["--quality", quality]
    if frames:
        extra += ["--frames", frames]
    if overwrite:
        extra.append("--overwrite")
    cmd = _worker_cmd("preview" if quality == "preview" else "render", d, extra)

    job = RenderJob(id=uuid.uuid4().hex, project_id=project_id, op="render", quality=quality)
    with _lock:
        _jobs[job.id] = job
    threading.Thread(target=_run, args=(job, d, cmd), daemon=True, name=f"video-render-{job.id[:6]}").start()
    return job


def register_job(job: RenderJob) -> None:
    with _lock:
        _jobs[job.id] = job


def active_job_for(project_id: str, ops: tuple[str, ...]) -> Optional[RenderJob]:
    with _lock:
        return next((j for j in _jobs.values() if j.project_id == project_id and j.op in ops and j.status in (QUEUED, RUNNING)), None)


def get_job(job_id: str) -> Optional[RenderJob]:
    with _lock:
        return _jobs.get(job_id)


def cancel_job(job_id: str) -> bool:
    job = get_job(job_id)
    if not job or job.status in _TERMINAL:
        return False
    job._cancel.set()
    if job._task is not None:
        job._task.cancel()
    proc = job._proc
    if proc and proc.poll() is None:
        try:
            assert proc.stdin
            proc.stdin.write("cancel\n")
            proc.stdin.flush()
        except (OSError, ValueError, AssertionError):
            pass
        # Worker should exit on its own; escalate if it does not.
        threading.Timer(10, _kill_if_alive, args=(proc,)).start()
    return True


def _kill_if_alive(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()


def _run(job: RenderJob, project_dir: Path, cmd: list[str]) -> None:
    job.message = "Waiting for a render slot"
    while not _slots.acquire(timeout=0.25):
        if job._cancel.is_set():
            _finish(job, CANCELLED, error={"code": "cancelled", "message": "cancelled while queued"})
            return
    log_path = project_dir / "logs" / f"render-{job.id[:8]}.log"
    try:
        if job._cancel.is_set():
            _finish(job, CANCELLED, error={"code": "cancelled", "message": "cancelled while queued"})
            return
        job.status, job.stage, job.message = RUNNING, "starting", "Starting renderer"
        projects.set_status(job.project_id, "rendering")
        env = {**os.environ, "NODE_NO_WARNINGS": "1"}
        with open(log_path, "w", encoding="utf-8") as log:
            try:
                job._proc = subprocess.Popen(
                    cmd, cwd=RENDERER_DIR, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                )
            except OSError as exc:
                _finish(job, FAILED, error={"code": "spawn_failed", "message": str(exc)})
                return
            final: Optional[dict[str, Any]] = None
            assert job._proc.stdout
            for line in job._proc.stdout:
                if '"event":"progress"' not in line:
                    log.write(line)  # progress is high-frequency; keep the log readable
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = ev.get("event")
                if kind == "progress":
                    job.stage = ev.get("stage", job.stage)
                    job.progress = float(ev.get("progress", job.progress))
                    job.message = ev.get("message", job.message)
                elif kind in ("result", "error"):
                    final = ev
            code = job._proc.wait()

        if final and final.get("event") == "result" and code == 0:
            _finish(job, COMPLETED, result=final)
        elif final and final.get("code") == "cancelled" or job._cancel.is_set():
            _finish(job, CANCELLED, error=final or {"code": "cancelled", "message": "cancelled"})
        else:
            _finish(job, FAILED, error=final or {"code": "worker_crashed", "message": f"worker exited with code {code}; see {log_path.name}"})
    except Exception as exc:  # never let a render thread die silently
        _finish(job, FAILED, error={"code": "internal", "message": str(exc)})
    finally:
        _slots.release()


def finish_job(job: RenderJob, status: str, result: Optional[dict] = None, error: Optional[dict] = None) -> None:
    _finish(job, status, result, error)


def _finish(job: RenderJob, status: str, result: Optional[dict] = None, error: Optional[dict] = None) -> None:
    job.status, job.result, job.error, job.finished_at = status, result, error, time.time()
    if status == COMPLETED:
        job.progress, job.stage = 1.0, "done"
    try:
        # A failed/cancelled render never touches the previous output or the storyboard.
        projects.set_status(job.project_id, "rendered" if status == COMPLETED and job.quality == "production" else "draft")
    except Exception:
        pass
