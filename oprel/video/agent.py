"""Video Studio agent: brief -> plan -> narration -> per-scene code (write, check, fix, rerun) -> assembled storyboard -> render.

The agent is a long-running, persisted and resumable background run (``<project>/agent/run.json`` + ``events.jsonl``).
It only ever uses existing Oprel services for models and speech. Scene code written by the model is data until the
render worker has applied the sandbox policy (static checks, TypeScript, network-isolated browser) to it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

from oprel.video import ai, projects, renderer_jobs

PROMPTS_DIR = renderer_jobs.RENDERER_DIR / "prompts"
FPS = 30
WORDS_PER_SECOND = 2.4

ACTIVE_STATUSES = {"planning", "narrating", "writing", "assembling", "rendering"}
RESUMABLE_STATUSES = {"interrupted", "needs_attention", "failed", "cancelled"}

DEFAULTS: dict[str, Any] = {
    "target_seconds": 45,
    "max_attempts": 4,          # per scene: write + up to 3 fixes
    "plan_attempts": 3,
    "time_budget_s": 4 * 3600,  # whole run
    "model_timeout_s": 1500,
    "check_timeout_s": 900,
    "auto_render": True,
    "review_plan": False,
    "allow_cloud": False,
    "tts_model": "",
    "voice": "",
}

MAX_MODEL_TOKENS_PLAN = 3000
MAX_MODEL_TOKENS_CODE = 6000

# Context windows (tokens). Below CONTEXT_COMPACT the scene prompt plus a usable amount of code cannot fit at all.
CONTEXT_FULL = 12000      # full prompts: API doc + worked example
CONTEXT_COMPACT = 3500    # compact prompts: short rules, no example
MIN_PLAN_ROOM = 900       # tokens that must remain for the model's plan
MIN_CODE_ROOM = 1200      # tokens that must remain for the model's scene file


class AgentStop(Exception):
    """The run cannot continue but can be resumed (budget exhausted, scene needs attention, ...)."""

    def __init__(self, message: str, code: str = "needs_attention"):
        super().__init__(message)
        self.code = code


class AgentCancelled(Exception):
    pass


class _Active:
    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.procs: list = []
        self.task: Optional[asyncio.Task] = None


_active: dict[str, _Active] = {}


# ── persistence ──────────────────────────────────────────────────────────────

def _agent_dir(pid: str) -> Path:
    d = projects.project_dir(pid) / "agent"
    d.mkdir(exist_ok=True)
    return d


def _save(pid: str, run: dict[str, Any]) -> None:
    run["updated_at"] = time.time()
    f = _agent_dir(pid) / "run.json"
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, f)


def _load(pid: str) -> Optional[dict[str, Any]]:
    f = projects.project_dir(pid) / "agent" / "run.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _event(pid: str, run: dict[str, Any], message: str, level: str = "info", phase: Optional[str] = None, scene: Optional[str] = None) -> None:
    run["seq"] = run.get("seq", 0) + 1
    ev = {"seq": run["seq"], "ts": time.time(), "level": level, "phase": phase or run.get("phase"), "scene": scene, "message": message}
    with open(_agent_dir(pid) / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def read_events(pid: str, since: int = 0, limit: int = 300) -> list[dict[str, Any]]:
    f = projects.project_dir(pid) / "agent" / "events.jsonl"
    if not f.is_file():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("seq", 0) > since:
            out.append(ev)
    return out[-limit:]


PLAN_SYSTEM_EXTRACT = """You are a video script compiler and motion graphics director.
The user has provided a custom scene-by-scene script with scenes, narration, timings in seconds, and visual directions.
Your job is to EXTRACT their exact scenes, narration, timings, and visual motion descriptions into the required plain-text format.
Do NOT invent new scenes or alter the user's narrative. Keep their exact words and timing seconds.

Reply ONLY in this plain-text format (no markdown, no JSON):

TITLE: <video title>
LOGLINE: <one sentence: what the viewer understands by the end>

SCENE 1: <scene title, max 6 words>
narration: <the exact narration words the narrator says for scene 1>
visual: <motion graphics visual description: motion, curves, colors, layout>
seconds: <exact seconds from user script, e.g. 6>

SCENE 2: <scene title>
narration: ...
visual: ...
seconds: ...

Guidelines:
- Preserve the user's scene breakdown, timings in seconds, and exact narration verbatim.
- For each scene's visual: describe high-impact MOTION GRAPHICS (kinetic typography, animated vector paths, node networks, dynamic meters, 3D meshes). No static application wireframes or UI cards.
- Narration is plain spoken text: no markdown, no scene numbers, no stage directions, no timings like "10 seconds".
"""

PLAN_SYSTEM = """You are the director and scriptwriter for short educational explainer videos about AI, made for phones (vertical 9:16, YouTube Shorts).
You plan the video scene by scene. Another process then writes the code for each scene, so every scene's "visual" must be concrete enough to build
from simple shapes, text, diagrams and 3D primitives (there are no photos, stock footage or image files).

Reply ONLY in this plain-text format (no markdown, no JSON):

TITLE: <video title>
LOGLINE: <one sentence: what the viewer understands by the end>

SCENE 1: <scene title, max 6 words>
narration: <the exact words the narrator says: natural spoken English, 20-45 words>
visual: <3-5 sentences: what appears on screen, how it moves, where it sits (top/middle), colors, any diagram or metaphor>
seconds: <rough length in seconds>

SCENE 2: <scene title>
narration: ...
visual: ...
seconds: ...

Guidelines:
- About {scenes} scenes for a {seconds}-second video (about 2.4 spoken words per second overall).
- Open with a hook that creates curiosity. End with a short recap or takeaway.
- Design high-impact MOTION GRAPHICS: kinetic typography, pulsing node networks, animated data curves, rotating 3D geometry, radar charts, and glowing particle flows. Do NOT describe static UI cards, website wireframes, or rectangular application boxes with numbered labels.
- One idea per scene. Vary the visual metaphors (data flows, radar charts, 3D objects, kinetic text). Do not repeat the same layout twice in a row.
- Technically accurate. Never invent statistics, benchmarks, citations or quotes. If a visual shows example numbers, say they are illustrative.
- Narration is plain spoken text: no markdown, no scene numbers, no stage directions, no timings like "10 seconds".
"""

PLAN_SYSTEM_COMPACT = """You plan short vertical explainer videos about AI. Reply ONLY in this plain-text format:

TITLE: <title>
LOGLINE: <one sentence>

SCENE 1: <scene title>
narration: <20-40 spoken words>
visual: <2-4 sentences: what appears on screen and how it moves>
seconds: <length>

(repeat SCENE 2, 3, ...)

About {scenes} scenes for {seconds} seconds (2.4 spoken words per second). Start with a hook, end with a takeaway.
No invented statistics. Narration is plain spoken text: no markdown, no timings.
"""

_SCENE_HEAD = re.compile(r"^\W*scene\s*(\d+)\s*[:.\-)]?\s*(.*)$", re.I)
_FIELD = re.compile(r"^\W*(title|logline|narration|visual|seconds)\s*:\s*(.*)$", re.I)


def _clean(v: str) -> str:
    return re.sub(r"[*`#]+", "", v).strip().strip('"').strip()


def _slug(text: str, n: int = 24) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n].strip("-") or "scene"


def parse_plan(text: str, target_seconds: int) -> tuple[dict[str, Any], list[dict]]:
    """Parse the director's plain-text plan. Returns (plan, problems); problems are fed back to the model."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    title = logline = ""
    scenes: list[dict[str, Any]] = []
    cur: Optional[dict[str, Any]] = None
    last_field: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SCENE_HEAD.match(line)
        if m:
            cur = {"title": _clean(m.group(2)), "narration": "", "visual": "", "seconds": ""}
            scenes.append(cur)
            last_field = "title"
            continue
        f = _FIELD.match(line)
        if f:
            key, val = f.group(1).lower(), _clean(f.group(2))
            if cur is None:
                if key == "title":
                    title, last_field = val, "ptitle"
                elif key == "logline":
                    logline, last_field = val, "plogline"
                continue
            if key == "title":
                cur["title"] = val or cur["title"]
            else:
                cur[key] = val
            last_field = key
            continue
        # continuation line of a wrapped value
        if cur is not None and last_field in ("narration", "visual"):
            cur[last_field] = f"{cur[last_field]} {_clean(line)}".strip()
        elif cur is None and last_field == "plogline":
            logline = f"{logline} {_clean(line)}".strip()

    problems: list[dict] = []

    def bad(path: str, msg: str) -> None:
        problems.append({"level": "error", "path": path, "message": msg})

    if not scenes:
        return {}, [{"level": "error", "path": "", "message": "no 'SCENE <n>: <title>' blocks found; follow the plan format exactly"}]
    if not title:
        bad("TITLE", "missing 'TITLE:' line")
    if not 3 <= len(scenes) <= 14:
        bad("scenes", f"plan has {len(scenes)} scenes; use 3 to 14")
    total_words = 0
    out_scenes = []
    for i, s in enumerate(scenes):
        label = f"scene {i + 1}"
        words = len(s["narration"].split())
        total_words += words
        if not s["title"]:
            bad(label, "missing scene title after 'SCENE n:'")
        if words < 8:
            bad(label, f"narration is too short ({words} words); write 15-45 words of real spoken sentences")
        elif words > 90:
            bad(label, f"narration is too long ({words} words); keep each scene under 60 words")
        if re.search(r"\b\d+\s*(?:seconds?|secs?)\b", s["narration"], re.I):
            bad(label, "narration must not mention timings like '10 seconds'; write only what is spoken")
        if len(s["visual"]) < 25:
            bad(label, "'visual:' must describe concretely what appears on screen (3-5 sentences)")
        try:
            secs = float(re.findall(r"[0-9]*\.?[0-9]+", s["seconds"])[0]) if s["seconds"] else 0.0
        except (IndexError, ValueError):
            secs = 0.0
        sid = f"s{i + 1:02d}-{_slug(s['title'])}"
        out_scenes.append({"id": sid, "title": s["title"][:80], "narration": s["narration"][:700], "visual": s["visual"][:1200], "seconds": secs})
    want = target_seconds * WORDS_PER_SECOND
    if total_words and not problems and not 0.5 * want <= total_words <= 1.6 * want:
        bad("narration", f"total narration is {total_words} words, but a {target_seconds}-second video needs about {int(want)}; "
                         f"{'add' if total_words < want else 'cut'} narration")
    return {"title": title[:150], "logline": logline[:300], "scenes": out_scenes}, problems


def extract_code(reply: str) -> Optional[str]:
    """Pull the scene source out of a model reply (fenced block preferred)."""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S | re.I)
    blocks = re.findall(r"```(?:tsx|typescript|ts|jsx|javascript|js)?[ \t]*\n(.*?)```", reply, flags=re.S | re.I)
    blocks = [b for b in blocks if "export default" in b] or blocks
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    if "export default" in reply and "import " in reply:
        return reply.strip() + "\n"
    return None


# ── prompts ──────────────────────────────────────────────────────────────────

def _scene_system(compact: bool = False) -> str:
    if compact:  # small context window: short rules without prescriptive examples
        return (PROMPTS_DIR / "scene_api_compact.md").read_text(encoding="utf-8")
    return (PROMPTS_DIR / "scene_api.md").read_text(encoding="utf-8")


def _scene_user(plan: dict, idx: int, st: dict[str, Any], scene: dict[str, Any]) -> str:
    scenes = plan["scenes"]
    prev = scenes[idx - 1]["title"] if idx > 0 else "(none, this is the opening scene)"
    nxt = scenes[idx + 1]["title"] if idx + 1 < len(scenes) else "(none, this is the final scene)"
    frames = st["duration_frames"]
    extra = f"\nExtra direction from the user: {st['feedback']}\n" if st.get("feedback") else ""
    return (
        f'Video: "{plan["title"]}". {plan.get("logline", "")}\n'
        f'This is scene {idx + 1} of {len(scenes)}: "{scene["title"]}".\n'
        f'Spoken narration for context (DO NOT render this text on screen; subtitles are added automatically by the player): "{scene["narration"]}"\n'
        f'What the viewer should see: {scene["visual"]}\n'
        f"Scene length: {frames} frames ({frames / FPS:.1f}s at {FPS} fps).\n"
        f"Maintain continuous, frame-accurate animation across the entire {frames}-frame timeline: "
        f"orchestrate an energetic entrance over the first 15-20% of frames, sustained secondary micro-motion (subtle drift, rotations, pulsing paths, live counters) across the main body, and a clean resolution or transition-out in the final 10-15%. Do NOT freeze or stop moving after a few frames.\n"
        "DO NOT render subtitles, paragraphs of narration, or caption text in your JSX (the video player renders captions globally). Only render kinetic headlines, visual diagrams, telemetry, and motion graphics.\n"
        f"Previous scene: {prev}. Next scene: {nxt}. Design unique visuals tailored specifically to this scene's message.{extra}\n"
        "Write the scene file now using pure Remotion skills."
    )


def _fix_user(base_user: str, code: Optional[str], issues: list[dict], stage: str, repeated: bool = False) -> str:
    nag = (
        "\nIMPORTANT: your previous attempt failed with the same error. Do not resubmit the same code. Change the lines the problems point at, "
        "following the hints exactly.\n" if repeated else ""
    )
    lines = "\n".join(f"- {i.get('path') or 'file'}: {i.get('message')}" for i in issues[:14])
    previous = f"\n\nYour previous file:\n```tsx\n{code}```" if code else "\n\n(Your previous file is omitted to save space; write the whole file again.)"
    focus = _problem_lines(code, issues) if code else ""
    return (
        f"{base_user}{previous}{focus}\n\n"
        f"The automated {stage} check found these problems:\n{lines}\n{nag}\n"
        "Fix every problem. Reply with the complete corrected file in one ```tsx code block."
    )


def _problem_lines(code: str, issues: list[dict]) -> str:
    """The exact source lines the checker pointed at, numbered, so a small model sees what to change instead of re-reading the whole file."""
    lines = code.splitlines()
    picked: dict[int, str] = {}
    for i in issues[:8]:
        m = re.search(r":(\d+)$", str(i.get("path", "")))
        if m and 1 <= int(m.group(1)) <= len(lines):
            n = int(m.group(1))
            picked[n] = lines[n - 1].strip()[:160]
    if not picked:
        return ""
    return "\n\nThe lines to change (line number: text):\n" + "\n".join(f"{n}: {t}" for n, t in sorted(picked.items()))


def _prompt_mode(context: Optional[int]) -> str:
    """full / compact by the model's real context window (unknown, e.g. cloud, counts as large)."""
    if context is None or context >= CONTEXT_FULL:
        return "full"
    return "compact"


def _budget(run: dict[str, Any], system: str, user: str, want: int, min_room: int, what: str) -> int:
    """Max tokens to request, or a clear stop when the prompt leaves too little room in the model's context window."""
    ctx = run.get("context_tokens")
    if not ctx:
        return want
    used = ai.estimate_tokens(system) + ai.estimate_tokens(user) + 40  # + chat template overhead
    room = ctx - used
    if room < min_room:
        raise AgentStop(_small_context_message(ctx, used + min_room, what), "small_context")
    return max(256, min(want, room - 16))


def _small_context_message(ctx: int, needed: int, what: str) -> str:
    return (
        f"The model is running with a {ctx}-token context window, but {what} needs about {needed}. "
        "Set a larger context in Model memory (8192 or more works well), save and reload the model, then Resume."
    )


# ── the run ──────────────────────────────────────────────────────────────────

def _new_run(config: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    return {
        "version": 1, "status": "planning", "phase": "plan", "config": config, "plan": None, "plan_approved": False,
        "scenes": {}, "narrated": False, "message": "", "error": None, "seq": 0, "started_at": now, "updated_at": now,
        "finished_at": None, "render": None,
    }


def _set(pid: str, run: dict[str, Any], status: str, phase: Optional[str] = None, message: str = "") -> None:
    run["status"] = status
    if phase:
        run["phase"] = phase
    run["message"] = message
    _save(pid, run)


def _check_alive(run: dict[str, Any], act: _Active) -> None:
    if act.cancel.is_set():
        raise AgentCancelled()
    if time.time() - run["started_at"] > run["config"]["time_budget_s"]:
        raise AgentStop("the run's time budget is used up; resume to continue where it stopped")


async def _ask(pid: str, run: dict[str, Any], act: _Active, system: str, user: str, max_tokens: int, what: str, scene: Optional[str] = None,
               min_room: int = MIN_CODE_ROOM, temperature: float = 0.4) -> str:
    _check_alive(run, act)
    max_tokens = _budget(run, system, user, max_tokens, min_room, what)
    t0 = time.time()
    try:
        text = await asyncio.wait_for(ai._ask_model(run["config"]["model_id"], system, user, max_tokens, temperature), timeout=run["config"]["model_timeout_s"])
    except asyncio.TimeoutError:
        raise AgentStop(f"the model did not answer within {run['config']['model_timeout_s']}s ({what})")
    except Exception as exc:
        ctx_err = ai.parse_context_error(str(exc))
        if ctx_err:  # the backend rejected the prompt: report the real numbers instead of a raw backend error
            requested, available = ctx_err
            run["context_tokens"] = available
            raise AgentStop(_small_context_message(available, requested, what), "small_context")
        raise
    _event(pid, run, f"{what}: model replied ({len(text)} chars, {time.time() - t0:.0f}s)", scene=scene)
    _save(pid, run)
    _check_alive(run, act)
    return text


async def _phase_plan(pid: str, run: dict[str, Any], act: _Active) -> None:
    if run.get("plan"):
        return
    cfg = run["config"]
    _set(pid, run, "planning", "plan", "Planning the video")
    _event(pid, run, "Planning the video scene by scene")
    brief = cfg["brief"]
    has_structured_scenes = bool(
        re.search(r"\bscene\s*\d+\b", brief, re.I)
        and (
            "narration" in brief.lower()
            or "visual" in brief.lower()
            or "seconds" in brief.lower()
            or re.search(r"\(\s*\d+\s*s(?:ecs?)?\s*\)", brief, re.I)
        )
    )

    if has_structured_scenes:
        system = PLAN_SYSTEM_EXTRACT
        user = f"The user has provided a custom scene-by-scene script with explicit scenes, narration, timings in seconds, and visual directions.\nExtract and compile their exact scenes, seconds, narration, and motion directions into the required plain-text format.\n\nUser Script & Scenes:\n{brief}"
        _event(pid, run, "Detected custom script with scenes and timings: compiling scenes and motion directions")
    else:
        template = PLAN_SYSTEM if _prompt_mode(run.get("context_tokens")) == "full" else PLAN_SYSTEM_COMPACT
        system = template.replace("{scenes}", str(max(4, round(cfg["target_seconds"] / 6.5)))).replace("{seconds}", str(cfg["target_seconds"]))
        user = f"Plan a video of about {cfg['target_seconds']} seconds.\n\nWhat the user wants:\n{brief}"
    reply, problems, plan = "", [], None
    for attempt in range(cfg["plan_attempts"]):
        prompt = user if attempt == 0 else (
            f"{user}\n\nYour previous plan:\n{reply[:6000]}\n\nProblems found:\n"
            + "\n".join(f"- {p['path']}: {p['message']}" for p in problems) + "\nFix them and reply with the complete corrected plan in the same format."
        )
        if attempt and ai.estimate_tokens(system) + ai.estimate_tokens(prompt) + MIN_PLAN_ROOM > (run.get("context_tokens") or 10**9):
            prompt = f"{user}\n\nYour previous plan had these problems:\n" + "\n".join(f"- {p['path']}: {p['message']}" for p in problems) + "\nWrite the whole plan again, fixing them."
        reply = await _ask(pid, run, act, system, prompt, MAX_MODEL_TOKENS_PLAN, "Plan" if attempt == 0 else f"Plan (revision {attempt})", min_room=MIN_PLAN_ROOM)
        plan, problems = parse_plan(reply, cfg["target_seconds"])
        if not problems:
            break
        _event(pid, run, f"Plan needs changes: {problems[0]['message']}" + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""), level="warn")
    if problems or not plan:
        raise AgentStop("the model could not produce a usable plan: " + "; ".join(p["message"] for p in problems[:3]))
    run["plan"] = plan
    run["scenes"] = {s["id"]: _scene_state(s, cfg) for s in plan["scenes"]}
    run["narrated"] = False
    _event(pid, run, f"Plan ready: \"{plan['title']}\" with {len(plan['scenes'])} scenes")
    _save(pid, run)


def _scene_state(scene: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    words = len(scene["narration"].split())
    secs = scene.get("seconds") or max(3.0, words / WORDS_PER_SECOND + 1.0)
    return {"status": "pending", "attempts": 0, "duration_frames": max(1, round(secs * FPS)), "file": f"scenes/{scene['id']}.tsx",
            "last_issues": [], "stills": [], "feedback": "", "history": []}


async def _phase_narrate(pid: str, run: dict[str, Any], act: _Active) -> None:
    cfg, plan = run["config"], run["plan"]
    if run.get("narrated"):
        return
    _set(pid, run, "narrating", "narrate", "Recording narration")
    if not cfg.get("tts_model"):
        _event(pid, run, "No narration model selected: timing comes from the plan and the video will have no voice-over", level="warn")
        run["narrated"] = True
        _save(pid, run)
        return
    _event(pid, run, f"Synthesizing narration with {cfg['tts_model']}")
    skeleton = {
        "schemaVersion": 1, "title": plan["title"], "format": {"preset": "shorts", "fps": FPS},
        "scenes": [{"id": s["id"], "type": "hook", "durationSeconds": 3, "narration": s["narration"], "params": {"headline": s["title"][:100] or "Scene"}} for s in plan["scenes"]],
    }
    projects.write_storyboard(pid, skeleton)

    def progress(_stage: str, value: float, message: str) -> None:
        run["message"] = message

    try:
        await asyncio.to_thread(ai.narrate_project, pid, cfg["tts_model"], cfg.get("voice") or None, 1.0, progress, act.cancel.is_set)
    except ai.AIStageError as exc:
        if exc.code == "cancelled":
            raise AgentCancelled()
        raise AgentStop(f"narration failed: {exc}")
    sb = projects.read_storyboard(pid)
    for s in sb["scenes"]:
        if s["id"] in run["scenes"]:
            run["scenes"][s["id"]]["duration_frames"] = int(s["durationFrames"])
    run["narrated"] = True
    total = sum(st["duration_frames"] for st in run["scenes"].values())
    _event(pid, run, f"Narration ready: video length {total / FPS:.1f}s, {len(sb.get('captions', []))} captions")
    _save(pid, run)


async def _phase_scenes(pid: str, run: dict[str, Any], act: _Active) -> None:
    cfg, plan = run["config"], run["plan"]
    compact = _prompt_mode(run.get("context_tokens")) == "compact"
    system = _scene_system(compact)
    if compact:
        _event(pid, run, "Small context window: using compact prompts (shorter rules, no worked example)", level="warn")
    d = projects.project_dir(pid)
    (d / "scenes").mkdir(exist_ok=True)
    (d / "agent" / "candidates").mkdir(parents=True, exist_ok=True)
    _set(pid, run, "writing", "scenes", "Writing scenes")
    for idx, scene in enumerate(plan["scenes"]):
        st = run["scenes"][scene["id"]]
        if st["status"] == "done":
            continue
        st["status"] = "writing"
        base_user = _scene_user(plan, idx, st, scene)
        code, issues, stage, last_first, last_reply = None, [], "", None, None
        _event(pid, run, f"Scene {idx + 1}/{len(plan['scenes'])} \"{scene['title']}\": writing", phase="scenes", scene=scene["id"])
        while st["attempts"] < cfg["max_attempts"]:
            _check_alive(run, act)
            st["attempts"] += 1
            n = st["attempts"]
            user = base_user
            if code is not None:
                first = (issues[0].get("message") if issues else None)
                repeated = bool(first) and first == last_first
                last_first = first
                user = _fix_user(base_user, code, issues, stage, repeated)
                ctx = run.get("context_tokens")
                if ctx and ai.estimate_tokens(system) + ai.estimate_tokens(user) + MIN_CODE_ROOM > ctx:
                    user = _fix_user(base_user, None, issues, stage, repeated)  # the old file would not leave room for the new one
            st["status"] = "writing" if code is None else "fixing"
            run["message"] = f"Scene {idx + 1}: attempt {n}"
            _save(pid, run)
            reply = await _ask(pid, run, act, system, user, MAX_MODEL_TOKENS_CODE, f"Scene {idx + 1} attempt {n}", scene["id"])
            new_code = extract_code(reply)
            # A model that ignores feedback resubmits the same file. Re-ask (same attempt) at a higher temperature instead of re-checking it.
            for temp in (0.8, 1.1):
                if new_code and new_code == last_reply:
                    _event(pid, run, f"Scene {idx + 1} attempt {n}: the model sent back the identical file; asking again with more variation", level="warn", scene=scene["id"])
                    reply = await _ask(pid, run, act, system, user, MAX_MODEL_TOKENS_CODE, f"Scene {idx + 1} attempt {n} (retry)", scene["id"], temperature=temp)
                    new_code = extract_code(reply)
            if new_code:
                last_reply = new_code
            if not new_code:
                issues, stage = [{"path": "reply", "message": "no code block found; reply with the complete file in one ```tsx block"}], "format"
                st["last_issues"] = issues
                _event(pid, run, f"Scene {idx + 1} attempt {n}: the reply had no code block", level="warn", scene=scene["id"])
                continue
            code = new_code
            cand_rel = f"agent/candidates/{scene['id']}-a{n}.tsx"
            (d / cand_rel).write_text(code, encoding="utf-8")
            st["status"] = "checking"
            _event(pid, run, f"Scene {idx + 1} attempt {n}: checking (policy, types, real render of sample frames)", scene=scene["id"])
            _save(pid, run)
            res = await asyncio.to_thread(
                renderer_jobs.run_worker, "check-scene", pid,
                ["--file", cand_rel, "--duration", str(st["duration_frames"]), "--narration", scene["narration"], "--title", scene["title"],
                 "--out", f"agent/stills/{scene['id']}/a{n}"],
                cfg["check_timeout_s"], act.cancel, act.procs,
            )
            if res.get("code") == "cancelled":
                raise AgentCancelled()
            if res.get("event") != "result":
                raise AgentStop(f"the scene checker could not run: {res.get('message') or res.get('code')}")
            fixed_rel = res.get("fixedFile")
            if fixed_rel:  # the checker mechanically repaired API slips; continue from that file so line numbers and hints match
                fixed_path = d / fixed_rel
                if fixed_path.is_file():
                    code = fixed_path.read_text(encoding="utf-8")
                    cand_rel = fixed_rel
                notes = ", ".join(sorted({f["rule"] for f in res.get("autofixes", [])}))
                _event(pid, run, f"Scene {idx + 1} attempt {n}: automatically repaired {len(res.get('autofixes', []))} API slip(s) ({notes})", scene=scene["id"])
            st["stills"] = [f["file"] for f in res.get("frames", [])]
            st["history"].append({"attempt": n, "file": cand_rel, "ok": bool(res.get("ok")), "stage": res.get("stage"), "issues": res.get("issues", [])[:8]})
            if res.get("ok"):
                final = d / st["file"]
                tmp = final.with_suffix(".tsx.tmp")
                shutil.copyfile(d / cand_rel, tmp)
                os.replace(tmp, final)
                st.update(status="done", last_issues=[], feedback="")
                note = f" (warning: {res['warnings'][0]['message']})" if res.get("warnings") else ""
                _event(pid, run, f"Scene {idx + 1} \"{scene['title']}\" passed all checks on attempt {n}{note}", scene=scene["id"])
                _save(pid, run)
                break
            issues, stage = res.get("issues", []), {"static": "safety/determinism", "types": "TypeScript", "render": "render"}.get(res.get("stage"), "render")
            st["last_issues"] = issues
            _event(pid, run, f"Scene {idx + 1} attempt {n} failed the {stage} check: {issues[0]['message'][:200]}" + (f" (+{len(issues) - 1} more)" if len(issues) > 1 else ""), level="warn", scene=scene["id"])
            _save(pid, run)
        else:
            st["status"] = "failed"
            _event(pid, run, f"Scene {idx + 1} \"{scene['title']}\" could not be fixed after {st['attempts']} attempts", level="error", scene=scene["id"])
            _save(pid, run)
    failed = [sid for sid, st in run["scenes"].items() if st["status"] != "done"]
    if failed:
        raise AgentStop(f"{len(failed)} scene(s) need attention: retry them with extra direction, or edit the plan")


async def _phase_assemble(pid: str, run: dict[str, Any], act: _Active) -> None:
    plan = run["plan"]
    _set(pid, run, "assembling", "assemble", "Assembling the storyboard")
    prev = projects.read_storyboard(pid) or {}
    has_audio = bool(prev.get("narration")) and run.get("narrated") and run["config"].get("tts_model")
    scenes = []
    for s in plan["scenes"]:
        st = run["scenes"][s["id"]]
        frames = st["duration_frames"]
        scenes.append({
            "id": s["id"], "type": "custom", "durationFrames": frames, "narration": s["narration"],
            "transition": {"type": "fade", "durationFrames": min(8, frames // 2)},
            "params": {"file": st["file"], "title": s["title"][:100]},
        })
    sb: dict[str, Any] = {"schemaVersion": 1, "title": plan["title"], "format": {"preset": "shorts", "fps": FPS}, "scenes": scenes}
    if has_audio:
        for k in ("narration", "captions", "captionsSource"):
            if prev.get(k):
                sb[k] = prev[k]
    issues = await asyncio.to_thread(ai._validate_candidate, pid, sb)
    if issues:
        raise AgentStop("the assembled storyboard did not validate: " + "; ".join(i["message"] for i in issues[:3]))
    projects.write_storyboard(pid, sb)
    _event(pid, run, f"Storyboard assembled: {len(scenes)} scenes, {sum(s['durationFrames'] for s in scenes) / FPS:.1f}s")
    _save(pid, run)


async def _phase_render(pid: str, run: dict[str, Any], act: _Active) -> None:
    _set(pid, run, "rendering", "render", "Rendering the final video")
    _event(pid, run, "Rendering the final 1080x1920 video")
    try:
        job = renderer_jobs.start_render(pid, "production", overwrite=True)
    except renderer_jobs.RendererUnavailable as exc:
        raise AgentStop(str(exc))
    run["render"] = {"job_id": job.id}
    _save(pid, run)
    while job.status in (renderer_jobs.QUEUED, renderer_jobs.RUNNING):
        if act.cancel.is_set():
            renderer_jobs.cancel_job(job.id)
            raise AgentCancelled()
        run["message"] = f"Rendering: {int(job.progress * 100)}% ({job.stage})"
        run["render"].update(progress=job.progress, stage=job.stage)
        _save(pid, run)
        await asyncio.sleep(2)
    run["render"].update(status=job.status, progress=job.progress)
    if job.status != renderer_jobs.COMPLETED:
        msg = (job.error or {}).get("message") or "render failed"
        issues = (job.error or {}).get("issues") or []
        run["render"]["error"] = job.error
        raise AgentStop(f"render failed: {msg}" + (f" ({issues[0]['path']}: {issues[0]['message']})" if issues else ""))
    _event(pid, run, "Final video rendered and validated")


async def _detect_context(pid: str, run: dict[str, Any]) -> None:
    ctx = await ai.model_context_tokens(run["config"]["model_id"])
    run["context_tokens"] = ctx
    if ctx:
        mode = "full prompts" if _prompt_mode(ctx) == "full" else "compact prompts"
        _event(pid, run, f"Model context window: {ctx} tokens ({mode})", level="warn" if ctx < CONTEXT_COMPACT else "info")
        if ctx < CONTEXT_COMPACT:
            raise AgentStop(_small_context_message(ctx, CONTEXT_COMPACT + MIN_CODE_ROOM, "writing a scene"), "small_context")
    _save(pid, run)


async def _run(pid: str, act: _Active) -> None:
    run = _load(pid)
    assert run is not None
    run["error"] = None
    try:
        await _detect_context(pid, run)
        await _phase_plan(pid, run, act)
        if run["config"]["review_plan"] and not run.get("plan_approved"):
            _set(pid, run, "awaiting_review", "plan", "Review the plan, then continue")
            _event(pid, run, "Plan is ready for your review")
            return
        await _phase_narrate(pid, run, act)
        await _phase_scenes(pid, run, act)
        await _phase_assemble(pid, run, act)
        if run["config"]["auto_render"]:
            await _phase_render(pid, run, act)
        run["finished_at"] = time.time()
        _set(pid, run, "completed", "done", "Video is ready")
        _event(pid, run, f"Done in {int((run['finished_at'] - run['started_at']) / 60)} min", phase="done")
    except (AgentCancelled, asyncio.CancelledError):
        _set(pid, run, "cancelled", None, "Stopped by you; resume to continue")
        _event(pid, run, "Stopped by you", level="warn")
    except AgentStop as exc:
        run["error"] = {"code": exc.code, "message": str(exc)}
        _set(pid, run, "needs_attention", None, str(exc))
        _event(pid, run, str(exc), level="error")
    except Exception as exc:  # never let the run task die silently
        run["error"] = {"code": "internal", "message": f"{type(exc).__name__}: {exc}"}
        _set(pid, run, "failed", None, str(exc))
        _event(pid, run, f"Unexpected error: {type(exc).__name__}: {exc}", level="error")
    finally:
        _save(pid, run)
        _active.pop(pid, None)


# ── public API ───────────────────────────────────────────────────────────────

def _launch(pid: str) -> None:
    act = _Active()
    _active[pid] = act
    act.task = asyncio.get_running_loop().create_task(_run(pid, act))


def _ensure_idle(pid: str) -> None:
    if pid in _active:
        raise ai.AIStageError("busy", "an agent run is already active for this project")
    limit = max(1, int(os.environ.get("OPREL_VIDEO_MAX_AGENTS", "1")))
    if len(_active) >= limit:
        raise ai.AIStageError("busy", "another video is being produced right now; wait for it to finish or stop it")


def start_run(pid: str, brief: str, model_id: str, allow_code: bool, **options: Any) -> dict[str, Any]:
    projects.get_project(pid)
    brief = (brief or "").strip()
    if not brief or len(brief) > 4000:
        raise ai.AIStageError("bad_request", "describe the video in 1-4000 characters")
    if not allow_code:
        raise ai.AIStageError("consent_required", "the agent writes and runs scene code; confirm that you allow AI-written code to run (it is sandboxed and network-isolated)")
    cfg = {**DEFAULTS, **{k: v for k, v in options.items() if k in DEFAULTS and v is not None}, "brief": brief, "model_id": model_id}
    if not 15 <= int(cfg["target_seconds"]) <= 120:
        raise ai.AIStageError("bad_request", "target length must be 15-120 seconds")
    if not 1 <= int(cfg["max_attempts"]) <= 10:
        raise ai.AIStageError("bad_request", "attempts per scene must be 1-10")
    ai.check_model_request(model_id, bool(cfg["allow_cloud"]))
    if cfg["tts_model"] and not any(cfg["tts_model"].lower() in m["id"] or m["id"] in cfg["tts_model"].lower() for m in ai.tts_models_ready()):
        raise ai.AIStageError("tts_unavailable", "that narration model is not installed; download it on the Audio page or choose none")
    renderer = renderer_jobs.renderer_status()
    if not renderer["available"]:
        raise ai.AIStageError("renderer_unavailable", renderer["message"] or "renderer unavailable")
    _ensure_idle(pid)

    d = _agent_dir(pid)
    old = d / "run.json"
    if old.is_file():  # keep the previous run for reference
        hist = d / "history"
        hist.mkdir(exist_ok=True)
        shutil.move(str(old), str(hist / f"run-{int(time.time())}.json"))
    (d / "events.jsonl").unlink(missing_ok=True)
    run = _new_run(cfg)
    _save(pid, run)
    _event(pid, run, f"Started with model {model_id}; target {cfg['target_seconds']}s")
    _save(pid, run)
    _launch(pid)
    return get_run(pid)


def resume_run(pid: str) -> dict[str, Any]:
    run = _load(pid)
    if not run:
        raise ai.AIStageError("not_found", "no agent run for this project")
    status = get_run(pid)["run"]["status"]
    if status == "awaiting_review":
        raise ai.AIStageError("bad_request", "approve the plan with Continue")
    if status not in RESUMABLE_STATUSES:
        raise ai.AIStageError("bad_request", f"a run in state '{status}' cannot be resumed")
    ai.check_model_request(run["config"]["model_id"], bool(run["config"]["allow_cloud"]))
    _ensure_idle(pid)
    run["started_at"] = time.time()  # fresh time budget for the resumed segment
    for st in run.get("scenes", {}).values():
        if st["status"] in ("failed", "writing", "fixing", "checking"):
            st.update(status="pending", attempts=0)  # give unfinished scenes a fresh round of attempts
    run["status"], run["message"] = "planning" if not run.get("plan") else "writing", "Resuming"
    _save(pid, run)
    _event(pid, run, "Resumed")
    _save(pid, run)
    _launch(pid)
    return get_run(pid)


def approve_plan(pid: str) -> dict[str, Any]:
    run = _load(pid)
    if not run or get_run(pid)["run"]["status"] != "awaiting_review":
        raise ai.AIStageError("bad_request", "there is no plan waiting for review")
    _ensure_idle(pid)
    run["plan_approved"] = True
    run["started_at"] = time.time()
    _save(pid, run)
    _event(pid, run, "Plan approved")
    _save(pid, run)
    _launch(pid)
    return get_run(pid)


def cancel_run(pid: str) -> bool:
    act = _active.get(pid)
    if not act:
        return False
    act.cancel.set()
    for p in list(act.procs):
        try:
            if p.poll() is None:
                p.kill()
        except OSError:
            pass
    if act.task:
        act.task.cancel()
    return True


def update_plan(pid: str, plan_scenes: list[dict[str, Any]], title: Optional[str] = None) -> dict[str, Any]:
    """Edit narration/visual/title of existing scenes, or add/remove/reorder scenes, while the run is idle."""
    run = _load(pid)
    if not run or not run.get("plan"):
        raise ai.AIStageError("bad_request", "there is no plan to edit")
    if pid in _active:
        raise ai.AIStageError("busy", "stop the run before editing the plan")
    if not 1 <= len(plan_scenes) <= 14:
        raise ai.AIStageError("bad_request", "a plan needs 1-14 scenes")
    old = {s["id"]: s for s in run["plan"]["scenes"]}
    new_scenes, states, structure_changed = [], {}, False
    used: set[str] = set()
    for i, s in enumerate(plan_scenes):
        narration, visual, st_title = _clean(str(s.get("narration", ""))), _clean(str(s.get("visual", ""))), _clean(str(s.get("title", "")))
        if len(narration.split()) < 5 or len(visual) < 15 or not st_title:
            raise ai.AIStageError("bad_request", f"scene {i + 1} needs a title, narration and a visual description")
        sid = s.get("id") if s.get("id") in old else None
        if not sid:
            sid = f"s{i + 1:02d}-{_slug(st_title)}"
            while sid in used or sid in old:
                sid += "x"
            structure_changed = True
        used.add(sid)
        prev = old.get(sid)
        changed = prev is None or prev["narration"] != narration or prev["visual"] != visual or prev["title"] != st_title
        item = {"id": sid, "title": st_title[:80], "narration": narration[:700], "visual": visual[:1200], "seconds": prev["seconds"] if prev else 0.0}
        new_scenes.append(item)
        if prev and not changed:
            states[sid] = run["scenes"][sid]
        else:
            states[sid] = _scene_state(item, run["config"])
            if prev:
                states[sid]["feedback"] = run["scenes"][sid].get("feedback", "")
        if prev and prev["narration"] != narration:
            structure_changed = True
    if [s["id"] for s in new_scenes] != [s["id"] for s in run["plan"]["scenes"]]:
        structure_changed = True
    run["plan"]["scenes"] = new_scenes
    if title:
        run["plan"]["title"] = _clean(title)[:150]
    run["scenes"] = states
    if structure_changed:
        run["narrated"] = False  # audio and timings must be regenerated
    _save(pid, run)
    _event(pid, run, "Plan edited" + ("; narration will be regenerated" if structure_changed else ""))
    _save(pid, run)
    return get_run(pid)


def retry_scene(pid: str, scene_id: str, feedback: str = "") -> dict[str, Any]:
    run = _load(pid)
    if not run or scene_id not in run.get("scenes", {}):
        raise ai.AIStageError("not_found", "unknown scene")
    status = get_run(pid)["run"]["status"]
    if status not in RESUMABLE_STATUSES | {"completed"}:
        raise ai.AIStageError("busy", "stop the run before retrying a scene")
    _ensure_idle(pid)
    st = run["scenes"][scene_id]
    st.update(status="pending", attempts=0, last_issues=[], feedback=_clean(feedback)[:600])
    run["status"], run["message"], run["started_at"] = "writing", f"Retrying {scene_id}", time.time()
    _save(pid, run)
    _event(pid, run, f"Retrying scene {scene_id}" + (f" with direction: {feedback[:120]}" if feedback else ""), scene=scene_id)
    _save(pid, run)
    _launch(pid)
    return get_run(pid)


def get_run(pid: str, since: int = 0) -> dict[str, Any]:
    run = _load(pid)
    if run is None:
        return {"run": None, "events": [], "active": False}
    active = pid in _active
    if run["status"] in ACTIVE_STATUSES and not active:
        # the daemon restarted (or the task died) while this run was in progress
        run["status"], run["message"] = "interrupted", "The server restarted during this run; resume to continue"
        _save(pid, run)
        _event(pid, run, "Run was interrupted (server restart); you can resume it", level="warn")
        _save(pid, run)
    return {"run": run, "events": read_events(pid, since), "active": active}


def project_file(pid: str, rel: str) -> Optional[Path]:
    """Resolve a file the UI may display (scene stills, scene code, candidates). Anything else is refused."""
    if not re.fullmatch(r"(?:agent/stills/[A-Za-z0-9_./-]+\.png|agent/candidates/[A-Za-z0-9_.-]+\.tsx|scenes/[A-Za-z0-9_.-]+\.tsx)", rel) or ".." in rel:
        return None
    d = projects.project_dir(pid).resolve()
    f = (d / rel).resolve()
    if d not in f.parents or not f.is_file():
        return None
    return f
