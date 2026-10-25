from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from oprel.video import agent, ai, projects, renderer_jobs

router = APIRouter(tags=["Video Studio"])

_STATUS_BY_CODE = {"busy": 409, "consent_required": 403, "cloud_not_allowed": 403, "not_found": 404, "renderer_unavailable": 503}


def _raise(exc: ai.AIStageError) -> None:
    raise HTTPException(status_code=_STATUS_BY_CODE.get(exc.code, 400), detail=str(exc))


class CreateProjectRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    storyboard: Optional[dict[str, Any]] = None  # import an existing storyboard; normally omitted (the agent writes it)


class RenderRequest(BaseModel):
    quality: str = Field(default="production", pattern="^(preview|production)$")
    frames: Optional[str] = Field(default=None, pattern=r"^\d+-\d+$")
    overwrite: bool = False


class StartAgentRequest(BaseModel):
    brief: str = Field(..., min_length=1, max_length=4000)
    model_id: str = Field(..., min_length=1, max_length=200)
    allow_code: bool = False
    allow_cloud: bool = False
    target_seconds: int = Field(default=45, ge=15, le=120)
    tts_model: str = Field(default="", max_length=64)
    voice: str = Field(default="", max_length=64)
    review_plan: bool = False
    auto_render: bool = True
    max_attempts: int = Field(default=4, ge=1, le=10)
    time_budget_hours: float = Field(default=4, gt=0, le=48)


class PlanSceneModel(BaseModel):
    id: Optional[str] = Field(default=None, max_length=80)
    title: str = Field(..., max_length=120)
    narration: str = Field(..., max_length=900)
    visual: str = Field(..., max_length=1500)


class UpdatePlanRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=150)
    scenes: list[PlanSceneModel] = Field(..., min_length=1, max_length=14)


class RetrySceneRequest(BaseModel):
    feedback: str = Field(default="", max_length=600)


def _project_or_404(project_id: str) -> dict[str, Any]:
    try:
        return projects.get_project(project_id)
    except projects.ProjectError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/v1/video/status")
async def video_status():
    return renderer_jobs.renderer_status()


@router.get("/v1/video/tts-models")
async def list_video_tts_models():
    return {"data": [{"id": m["id"], "name": m["name"], "voices": m.get("voices", [])} for m in ai.tts_models_ready()]}


@router.get("/v1/video/projects")
async def list_video_projects():
    return {"data": projects.list_projects()}


@router.post("/v1/video/projects")
async def create_video_project(req: CreateProjectRequest):
    try:
        return projects.create_project(req.title, req.storyboard)
    except projects.ProjectError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/v1/video/projects/{project_id}")
async def get_video_project(project_id: str):
    meta = _project_or_404(project_id)
    return {**meta, "storyboard": projects.read_storyboard(project_id)}


@router.put("/v1/video/projects/{project_id}/storyboard")
async def put_storyboard(project_id: str, storyboard: dict[str, Any] = Body(...)):
    _project_or_404(project_id)
    projects.write_storyboard(project_id, storyboard)
    return {"ok": True}


@router.delete("/v1/video/projects/{project_id}")
async def delete_video_project(project_id: str):
    _project_or_404(project_id)
    if project_id in agent._active:
        raise HTTPException(status_code=409, detail="stop the running agent before deleting this project")
    projects.delete_project(project_id)
    return {"ok": True}


@router.post("/v1/video/projects/{project_id}/validate")
async def validate_video_project(project_id: str):
    _project_or_404(project_id)
    try:
        res = await asyncio.to_thread(renderer_jobs.run_blocking, "validate", project_id)
    except renderer_jobs.RendererUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"ok": res.get("event") == "result", **res}


# ── agent ────────────────────────────────────────────────────────────────────

@router.post("/v1/video/projects/{project_id}/agent/start")
async def start_agent(project_id: str, req: StartAgentRequest):
    _project_or_404(project_id)
    try:
        return agent.start_run(
            project_id, req.brief, req.model_id, req.allow_code,
            allow_cloud=req.allow_cloud, target_seconds=req.target_seconds, tts_model=req.tts_model, voice=req.voice,
            review_plan=req.review_plan, auto_render=req.auto_render, max_attempts=req.max_attempts,
            time_budget_s=int(req.time_budget_hours * 3600),
        )
    except ai.AIStageError as exc:
        _raise(exc)


@router.get("/v1/video/projects/{project_id}/agent")
async def get_agent(project_id: str, since: int = 0):
    _project_or_404(project_id)
    return agent.get_run(project_id, since)


@router.post("/v1/video/projects/{project_id}/agent/cancel")
async def cancel_agent(project_id: str):
    _project_or_404(project_id)
    return {"cancelled": agent.cancel_run(project_id)}


@router.post("/v1/video/projects/{project_id}/agent/resume")
async def resume_agent(project_id: str):
    _project_or_404(project_id)
    try:
        return agent.resume_run(project_id)
    except ai.AIStageError as exc:
        _raise(exc)


@router.post("/v1/video/projects/{project_id}/agent/approve")
async def approve_agent_plan(project_id: str):
    _project_or_404(project_id)
    try:
        return agent.approve_plan(project_id)
    except ai.AIStageError as exc:
        _raise(exc)


@router.put("/v1/video/projects/{project_id}/agent/plan")
async def update_agent_plan(project_id: str, req: UpdatePlanRequest):
    _project_or_404(project_id)
    try:
        return agent.update_plan(project_id, [s.model_dump() for s in req.scenes], req.title)
    except ai.AIStageError as exc:
        _raise(exc)


@router.post("/v1/video/projects/{project_id}/agent/scenes/{scene_id}/retry")
async def retry_agent_scene(project_id: str, scene_id: str, req: RetrySceneRequest = RetrySceneRequest()):
    _project_or_404(project_id)
    try:
        return agent.retry_scene(project_id, scene_id, req.feedback)
    except ai.AIStageError as exc:
        _raise(exc)


@router.get("/v1/video/projects/{project_id}/files/{path:path}")
async def get_project_file(project_id: str, path: str):
    """Scene stills (png) and scene code (tsx) for the Video Studio UI. Nothing else is served."""
    _project_or_404(project_id)
    f = agent.project_file(project_id, path)
    if f is None:
        raise HTTPException(status_code=404, detail="file not found")
    media = "image/png" if f.suffix == ".png" else "text/plain; charset=utf-8"
    return FileResponse(f, media_type=media, headers={"Cache-Control": "no-cache"})


# ── rendering ────────────────────────────────────────────────────────────────

@router.post("/v1/video/projects/{project_id}/render")
async def render_video_project(project_id: str, req: RenderRequest = RenderRequest()):
    _project_or_404(project_id)
    if project_id in agent._active:
        raise HTTPException(status_code=409, detail="the agent is working on this project")
    try:
        job = renderer_jobs.start_render(project_id, req.quality, req.frames, req.overwrite)
    except renderer_jobs.RendererUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:  # includes ProjectError
        raise HTTPException(status_code=400, detail=str(exc))
    return job.public()


@router.get("/v1/video/jobs/{job_id}")
async def get_video_job(job_id: str):
    job = renderer_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job.public()


@router.post("/v1/video/jobs/{job_id}/cancel")
async def cancel_video_job(job_id: str):
    if not renderer_jobs.get_job(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    return {"cancelled": renderer_jobs.cancel_job(job_id)}


@router.get("/v1/video/projects/{project_id}/output/{name}")
async def get_video_output(project_id: str, name: str):
    _project_or_404(project_id)
    if name not in ("final.mp4", "preview.mp4"):
        raise HTTPException(status_code=404, detail="unknown output")
    f = projects.project_dir(project_id) / "output" / name
    if not f.is_file():
        raise HTTPException(status_code=404, detail="output not rendered yet")
    return FileResponse(f, media_type="video/mp4", filename=f"{project_id}-{name}")
