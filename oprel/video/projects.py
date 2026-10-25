"""Video project storage.

Layout (all under ``<cache_dir>/video_projects/<project_id>/``)::

    project.json        metadata (id, title, createdAt, updatedAt, status)
    storyboard.json     versioned storyboard (validated by the Node worker's Zod schema)
    assets/             narration, music, images (referenced by project-relative path only)
    output/             final.mp4, preview.mp4, *.validation.json
    logs/               one log file per render job
    tmp/                scratch space, safe to delete
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

PROJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
SUBDIRS = ("assets", "output", "logs", "tmp")

class ProjectError(ValueError):
    pass


def projects_root() -> Path:
    override = os.environ.get("OPREL_VIDEO_PROJECTS_DIR")
    if override:
        root = Path(override)
    else:
        from oprel.server.services.context import CONFIG

        root = Path(CONFIG.cache_dir) / "video_projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


def project_dir(project_id: str) -> Path:
    """Resolve a project id to its directory. Rejects anything that is not a plain slug."""
    if not isinstance(project_id, str) or not PROJECT_ID_RE.match(project_id):
        raise ProjectError("invalid project id")
    root = projects_root().resolve()
    p = (root / project_id).resolve()
    if p.parent != root:
        raise ProjectError("invalid project id")
    return p


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
    return s or "video"


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written storyboard


def create_project(title: str, storyboard: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    title = (title or "").strip()
    if not title or len(title) > 200:
        raise ProjectError("title must be 1-200 characters")
    pid = f"{_slug(title)}-{uuid.uuid4().hex[:6]}"
    d = project_dir(pid)
    d.mkdir(parents=True)
    for sub in SUBDIRS:
        (d / sub).mkdir()
    if storyboard is not None:  # an imported storyboard; otherwise the agent writes one
        sb = json.loads(json.dumps(storyboard))
        sb.setdefault("title", title)
        _write_json(d / "storyboard.json", sb)
    now = time.time()
    meta = {"id": pid, "title": title, "createdAt": now, "updatedAt": now, "status": "draft"}
    _write_json(d / "project.json", meta)
    return meta


def get_project(project_id: str) -> dict[str, Any]:
    d = project_dir(project_id)
    meta_file = d / "project.json"
    if not meta_file.is_file():
        raise ProjectError("project not found")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["hasFinalOutput"] = (d / "output" / "final.mp4").is_file()
    run_file = d / "agent" / "run.json"
    meta["agentStatus"] = None
    if run_file.is_file():
        try:
            meta["agentStatus"] = json.loads(run_file.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            pass
    return meta


def list_projects() -> list[dict[str, Any]]:
    out = []
    for child in projects_root().iterdir():
        if child.is_dir() and (child / "project.json").is_file():
            try:
                out.append(get_project(child.name))
            except (ProjectError, json.JSONDecodeError):
                continue
    return sorted(out, key=lambda m: m.get("updatedAt", 0), reverse=True)


def read_storyboard(project_id: str) -> Optional[dict[str, Any]]:
    """The project's storyboard, or None for a project the agent has not written one for yet."""
    d = project_dir(project_id)
    if not (d / "project.json").is_file():
        raise ProjectError("project not found")
    f = d / "storyboard.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def write_storyboard(project_id: str, storyboard: dict[str, Any]) -> None:
    d = project_dir(project_id)
    if not (d / "project.json").is_file():
        raise ProjectError("project not found")
    # Keep one previous version for undo / recovery.
    current = d / "storyboard.json"
    if current.is_file():
        shutil.copyfile(current, d / "storyboard.prev.json")
    _write_json(current, storyboard)
    set_status(project_id, "draft")


def set_status(project_id: str, status: str, **extra: Any) -> None:
    d = project_dir(project_id)
    meta = json.loads((d / "project.json").read_text(encoding="utf-8"))
    meta.update(status=status, updatedAt=time.time(), **extra)
    _write_json(d / "project.json", meta)


def delete_project(project_id: str) -> None:
    d = project_dir(project_id)
    if not (d / "project.json").is_file():
        raise ProjectError("project not found")
    shutil.rmtree(d)
