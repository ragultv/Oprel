"""Video Studio agent: plan parsing, run control and the write -> check -> fix loop.

Fast tests mock only the scene *checker*; the slow test at the bottom runs the real worker end to end
(real TypeScript check, real browser render of sample frames, real final render) with a scripted "model".
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import pytest

from oprel.video import agent, ai, projects, renderer_jobs

needs_renderer = pytest.mark.skipif(not renderer_jobs.renderer_status()["available"], reason="Node renderer deps not installed")
EXAMPLE = (renderer_jobs.RENDERER_DIR / "prompts" / "scene_example.tsx").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("OPREL_VIDEO_PROJECTS_DIR", str(tmp_path / "video_projects"))
    monkeypatch.setattr(ai, "is_cloud_model", lambda m: m.startswith("cloud::"))
    context = {"tokens": None}

    async def fake_context(model_id):
        return context["tokens"]

    monkeypatch.setattr(ai, "model_context_tokens", fake_context)  # never load a real model in tests
    agent._active.clear()
    return context


def narration(n: int = 18) -> str:
    return " ".join(["spoken"] * n) + "."


def plan_text(count: int = 3, words: int = 18, seconds: int = 3, title: bool = True) -> str:
    lines = ["TITLE: A test video" if title else "", "LOGLINE: Viewers learn how it works.", ""]
    for i in range(1, count + 1):
        lines += [
            f"SCENE {i}: Step number {i}",
            f"narration: {narration(words)}",
            "visual: A large diagram fades in at the top and a bright dot travels along three connected boxes in the middle of the screen.",
            f"seconds: {seconds}",
            "",
        ]
    return "\n".join(lines)


# ── pure parsing ─────────────────────────────────────────────────────────────

def test_parse_plan_builds_scene_ids_and_keeps_fields():
    plan, problems = agent.parse_plan(plan_text(4, words=30), target_seconds=45)
    assert problems == []
    assert [s["id"] for s in plan["scenes"]] == ["s01-step-number-1", "s02-step-number-2", "s03-step-number-3", "s04-step-number-4"]
    assert plan["title"] == "A test video" and plan["scenes"][0]["seconds"] == 3.0


def test_parse_plan_accepts_wrapped_lines_markdown_and_think_blocks():
    text = "<think>hm</think>\n**TITLE:** Wrapped\nLOGLINE: x\n" + "\n".join(
        f"## Scene {i}: Part {i}\n- **narration:** {narration(8)}\n  {narration(8)}\n- visual: {'A box moves across the screen and glows softly. ' * 2}\n" for i in (1, 2, 3)
    )
    plan, problems = agent.parse_plan(text, 20)
    assert [p for p in problems if "narration" not in p["path"]] == []
    assert len(plan["scenes"]) == 3 and len(plan["scenes"][0]["narration"].split()) == 16


@pytest.mark.parametrize("text,needle", [
    ("just chatting", "no 'SCENE"),
    (plan_text(2), "use 3 to 14"),
    (plan_text(3, words=5), "too short"),
    (plan_text(3).replace("spoken spoken", "ten seconds 10 seconds spoken", 1), "timings"),
    (plan_text(3, title=False), "TITLE"),
    (plan_text(3, words=80), "total narration"),
])
def test_parse_plan_reports_actionable_problems(text, needle):
    _, problems = agent.parse_plan(text, 45 if needle != "total narration" else 20)
    assert needle in " | ".join(p["message"] for p in problems)


def test_parse_plan_rejects_vague_visuals():
    text = plan_text(3).replace("A large diagram fades in at the top and a bright dot travels along three connected boxes in the middle of the screen.", "nice")
    _, problems = agent.parse_plan(text, 45)
    assert any("visual" in p["message"] for p in problems)


def test_extract_code_variants():
    code = "import React from 'react';\nexport default function S() { return null; }"
    assert agent.extract_code(f"Here you go:\n```tsx\n{code}\n```\nEnjoy") == code + "\n"
    assert agent.extract_code(f"```\n{code}\n```") == code + "\n"
    assert agent.extract_code(code) == code + "\n"
    assert agent.extract_code("```tsx\nconst a = 1;\n```\nand\n```tsx\n" + code + "\n```") == code + "\n"  # prefers the block with a default export
    assert agent.extract_code("sorry, I cannot do that") is None


# ── run control (no model, no worker) ────────────────────────────────────────

def start(pid, **kw):
    opts = dict(allow_code=True)
    opts.update(kw)
    return agent.start_run(pid, "explain RAG", "local-model", **opts)


def test_start_requires_code_consent_and_cloud_consent(monkeypatch):
    monkeypatch.setattr(renderer_jobs, "renderer_status", lambda: {"available": True, "message": None})
    pid = projects.create_project("gates")["id"]

    async def go():
        with pytest.raises(ai.AIStageError) as e:
            start(pid, allow_code=False)
        assert e.value.code == "consent_required"
        with pytest.raises(ai.AIStageError) as e:
            agent.start_run(pid, "x", "cloud::gpt", True)
        assert e.value.code == "cloud_not_allowed"
        with pytest.raises(ai.AIStageError) as e:
            agent.start_run(pid, "   ", "local-model", True)
        assert e.value.code == "bad_request"
        monkeypatch.setattr(ai, "tts_models_ready", lambda: [])
        with pytest.raises(ai.AIStageError) as e:
            start(pid, tts_model="kokoro-82m")
        assert e.value.code == "tts_unavailable"

    asyncio.run(go())


def test_start_fails_cleanly_when_renderer_missing(monkeypatch):
    monkeypatch.setattr(renderer_jobs, "renderer_status", lambda: {"available": False, "message": "no node"})
    pid = projects.create_project("norender")["id"]

    async def go():
        with pytest.raises(ai.AIStageError) as e:
            start(pid)
        assert e.value.code == "renderer_unavailable"

    asyncio.run(go())


def test_interrupted_runs_are_detected_and_resumable(monkeypatch):
    pid = projects.create_project("interrupt")["id"]
    run = agent._new_run({**agent.DEFAULTS, "brief": "b", "model_id": "m"})
    run["status"] = "writing"
    agent._save(pid, run)
    state = agent.get_run(pid)
    assert state["run"]["status"] == "interrupted" and not state["active"]
    assert any("interrupted" in e["message"] for e in state["events"])


def test_update_plan_marks_only_changed_scenes_and_regenerates_narration():
    pid = projects.create_project("edit plan")["id"]
    plan, _ = agent.parse_plan(plan_text(3), 45)
    cfg = {**agent.DEFAULTS, "brief": "b", "model_id": "m"}
    run = agent._new_run(cfg)
    run["plan"] = plan
    run["scenes"] = {s["id"]: agent._scene_state(s, cfg) for s in plan["scenes"]}
    for st in run["scenes"].values():
        st["status"] = "done"
    run["narrated"] = True
    run["status"] = "needs_attention"
    agent._save(pid, run)

    scenes = [dict(s) for s in plan["scenes"]]
    scenes[1]["visual"] = "A completely different animation with spinning rings in the middle of the frame."
    state = agent.update_plan(pid, scenes)
    sc = state["run"]["scenes"]
    assert sc[scenes[0]["id"]]["status"] == "done" and sc[scenes[1]["id"]]["status"] == "pending" and sc[scenes[2]["id"]]["status"] == "done"
    assert state["run"]["narrated"] is True  # visual-only edit keeps the audio

    scenes[2]["narration"] = narration(25)
    state = agent.update_plan(pid, scenes)
    assert state["run"]["narrated"] is False  # narration text changed: audio and timings are stale

    with pytest.raises(ai.AIStageError):
        agent.update_plan(pid, [{"title": "x", "narration": "no", "visual": "no"}])


def test_project_file_only_serves_whitelisted_paths():
    pid = projects.create_project("files")["id"]
    d = projects.project_dir(pid)
    (d / "agent" / "stills" / "s1").mkdir(parents=True)
    (d / "agent" / "stills" / "s1" / "f0.png").write_bytes(b"png")
    (d / "scenes").mkdir()
    (d / "scenes" / "a.tsx").write_text("x")
    (d / "project.json.bak").write_text("secret")
    assert agent.project_file(pid, "agent/stills/s1/f0.png") is not None
    assert agent.project_file(pid, "scenes/a.tsx") is not None
    for bad in ["project.json", "../project.json", "agent/run.json", "scenes/../project.json", "agent/stills/../../project.json", "scenes/missing.tsx", "output/final.mp4"]:
        assert agent.project_file(pid, bad) is None, bad


# ── the loop with a scripted model and a mocked checker ──────────────────────

def scripted_model(plan: str, per_scene: dict[int, list[str]], prompts: list[str]):
    """Returns a replacement for ai._ask_model: the plan first, then scene code by scene number and attempt."""
    attempts: dict[int, int] = {}

    async def fake(model_id, system, user, max_tokens, temperature=0.4):
        if "Plan a video" in user:
            return plan
        n = int(re.search(r"scene (\d+) of", user).group(1))
        prompts.append(user)
        i = attempts.get(n, 0)
        attempts[n] = i + 1
        replies = per_scene[n]
        return replies[min(i, len(replies) - 1)]

    return fake


def fenced(code: str) -> str:
    return f"```tsx\n{code}\n```"


def mock_checker(monkeypatch, verdicts: dict[str, list[dict]]):
    """Replace only the check-scene worker op; validate/other ops still run the real worker."""
    real = renderer_jobs.run_worker
    calls = []

    def fake(op, pid, extra=None, timeout=900, cancel=None, holder=None):
        if op != "check-scene":
            return real(op, pid, extra, timeout, cancel, holder)
        file = extra[extra.index("--file") + 1]
        scene_id = re.match(r"agent/candidates/(.+)-a\d+\.tsx", file).group(1)
        calls.append(file)
        queue = verdicts[scene_id]
        v = queue.pop(0) if len(queue) > 1 else queue[0]
        return {"event": "result", "ok": v["ok"], "stage": v.get("stage", "render"), "issues": v.get("issues", []), "warnings": [],
                "frames": [{"frame": 2, "file": f"agent/stills/{scene_id}/a1/f2.png"}], "exitCode": 0}

    monkeypatch.setattr(renderer_jobs, "run_worker", fake)
    return calls


def run_agent(pid, **kw):
    async def go():
        start(pid, auto_render=False, **kw)
        task = agent._active[pid].task
        await task
        return agent.get_run(pid)

    return asyncio.run(go())


@needs_renderer
def test_loop_fixes_a_failing_scene_feeds_errors_back_and_assembles(monkeypatch):
    pid = projects.create_project("loop")["id"]
    prompts: list[str] = []
    monkeypatch.setattr(ai, "_ask_model", scripted_model(plan_text(3), {1: [fenced("BROKEN"), fenced(EXAMPLE)], 2: [fenced(EXAMPLE)], 3: [fenced(EXAMPLE)]}, prompts))
    err = {"path": "agent/candidates/x.tsx:5", "message": "\"Date\" is not allowed: wall-clock time breaks deterministic rendering"}
    calls = mock_checker(monkeypatch, {"s01-step-number-1": [{"ok": False, "stage": "static", "issues": [err]}, {"ok": True}], "s02-step-number-2": [{"ok": True}], "s03-step-number-3": [{"ok": True}]})

    state = run_agent(pid)
    run = state["run"]
    assert run["status"] == "completed", run["error"]
    assert run["scenes"]["s01-step-number-1"]["attempts"] == 2
    fix_prompt = next(p for p in prompts if "Your previous file" in p)
    assert "wall-clock time" in fix_prompt and "BROKEN" in fix_prompt  # the checker's message and the old code reached the model
    assert len(calls) == 4
    sb = projects.read_storyboard(pid)
    assert [s["type"] for s in sb["scenes"]] == ["custom"] * 3
    assert all((projects.project_dir(pid) / s["params"]["file"]).is_file() for s in sb["scenes"])
    assert sum(s["durationFrames"] for s in sb["scenes"]) == 270
    msgs = " | ".join(e["message"] for e in state["events"])
    assert "passed all checks on attempt 2" in msgs and "failed the safety/determinism check" in msgs


@needs_renderer
def test_scene_that_never_passes_pauses_the_run_and_retry_with_feedback_finishes_it(monkeypatch):
    pid = projects.create_project("retry")["id"]
    prompts: list[str] = []
    monkeypatch.setattr(ai, "_ask_model", scripted_model(plan_text(3), {1: [fenced(EXAMPLE)], 2: [fenced(EXAMPLE)], 3: [fenced(EXAMPLE)]}, prompts))
    bad = {"ok": False, "stage": "render", "issues": [{"path": "render", "message": "the scene does not change over time"}]}
    mock_checker(monkeypatch, {"s01-step-number-1": [{"ok": True}], "s02-step-number-2": [bad], "s03-step-number-3": [{"ok": True}]})

    state = run_agent(pid, max_attempts=2)
    run = state["run"]
    assert run["status"] == "needs_attention"
    assert run["scenes"]["s02-step-number-2"]["status"] == "failed" and run["scenes"]["s02-step-number-2"]["attempts"] == 2
    assert run["scenes"]["s01-step-number-1"]["status"] == "done" and run["scenes"]["s03-step-number-3"]["status"] == "done"  # others were not blocked
    assert "need attention" in run["error"]["message"]

    mock_checker(monkeypatch, {"s02-step-number-2": [{"ok": True}]})

    async def go():
        agent.retry_scene(pid, "s02-step-number-2", "make the dot bigger")
        await agent._active[pid].task
        return agent.get_run(pid)

    state = asyncio.run(go())
    assert state["run"]["status"] == "completed", state["run"]["error"]
    assert any("make the dot bigger" in p for p in prompts)
    assert projects.read_storyboard(pid)["scenes"][1]["type"] == "custom"


@needs_renderer
def test_cancel_stops_the_run_and_resume_continues_without_redoing_finished_scenes(monkeypatch):
    pid = projects.create_project("cancel")["id"]
    prompts: list[str] = []
    gate = {"cancel_after_scene": 1}
    base = scripted_model(plan_text(3), {1: [fenced(EXAMPLE)], 2: [fenced(EXAMPLE)], 3: [fenced(EXAMPLE)]}, prompts)

    async def model(model_id, system, user, max_tokens, temperature=0.4):
        out = await base(model_id, system, user, max_tokens, temperature)
        if "scene 2 of" in user and gate["cancel_after_scene"]:
            agent.cancel_run(pid)
            await asyncio.sleep(0.05)
        return out

    monkeypatch.setattr(ai, "_ask_model", model)
    calls = mock_checker(monkeypatch, {f"s0{i}-step-number-{i}": [{"ok": True}] for i in (1, 2, 3)})

    state = run_agent(pid)
    assert state["run"]["status"] == "cancelled"
    assert state["run"]["scenes"]["s01-step-number-1"]["status"] == "done"
    done_before = len(calls)

    gate["cancel_after_scene"] = 0

    async def go():
        agent.resume_run(pid)
        await agent._active[pid].task
        return agent.get_run(pid)

    state = asyncio.run(go())
    assert state["run"]["status"] == "completed"
    assert not any("s01-step-number-1" in c for c in calls[done_before:])  # scene 1 was not rewritten


@needs_renderer
def test_plan_review_pauses_until_approved(monkeypatch):
    pid = projects.create_project("review")["id"]
    monkeypatch.setattr(ai, "_ask_model", scripted_model(plan_text(3), {i: [fenced(EXAMPLE)] for i in (1, 2, 3)}, []))
    mock_checker(monkeypatch, {f"s0{i}-step-number-{i}": [{"ok": True}] for i in (1, 2, 3)})
    state = run_agent(pid, review_plan=True)
    assert state["run"]["status"] == "awaiting_review" and state["run"]["plan"]["title"] == "A test video"
    assert not (projects.project_dir(pid) / "scenes").exists() or not list((projects.project_dir(pid) / "scenes").glob("*.tsx"))

    async def go():
        agent.approve_plan(pid)
        await agent._active[pid].task
        return agent.get_run(pid)

    assert asyncio.run(go())["run"]["status"] == "completed"


def test_unusable_plans_stop_the_run_with_the_models_problems(monkeypatch):
    monkeypatch.setattr(renderer_jobs, "renderer_status", lambda: {"available": True, "message": None})
    pid = projects.create_project("badplan")["id"]

    async def junk(*_a, **_k):
        return "I think videos are great!"

    monkeypatch.setattr(ai, "_ask_model", junk)
    state = run_agent(pid)
    assert state["run"]["status"] == "needs_attention" and "usable plan" in state["run"]["error"]["message"]
    assert state["run"]["plan"] is None


# ── real end-to-end: real checker, real browser, real final render ──────────

def good_scene(label: str) -> str:
    return EXAMPLE.replace('"How it works"', f'"{label}"')


@needs_renderer
@pytest.mark.slow
def test_real_pipeline_fixes_policy_type_and_static_scenes_then_renders(monkeypatch):
    pid = projects.create_project("real e2e")["id"]
    static_scene = (
        'import React from "react";\nimport { AbsoluteFill } from "remotion";\nimport { Backdrop, tokens } from "@oprel/kit";\n'
        'export default function Scene() {\n  return (<Backdrop accent="cyan"><AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>'
        '<div style={{ fontFamily: tokens.font.family, fontSize: 90, color: "white" }}>Static</div></AbsoluteFill></Backdrop>);\n}\n'
    )
    random_scene = good_scene("one").replace("const frame = useCurrentFrame();", "const frame = useCurrentFrame() + Math.random();")
    type_scene = good_scene("two").replace("const accent = tokens.accent.cyan;", "const accent = tokens.accent.cyan;\n  const broken: number = 'text';")
    prompts: list[str] = []
    replies = {1: [fenced(random_scene), fenced(good_scene("one"))], 2: [fenced(type_scene), fenced(good_scene("two"))], 3: [fenced(static_scene), fenced(good_scene("three"))]}
    monkeypatch.setattr(ai, "_ask_model", scripted_model(plan_text(3, seconds=3), replies, prompts))

    async def go():
        start(pid, auto_render=True)
        await agent._active[pid].task
        return agent.get_run(pid)

    state = asyncio.run(go())
    run = state["run"]
    assert run["status"] == "completed", run["error"]
    assert [run["scenes"][s]["attempts"] for s in run["scenes"]] == [2, 2, 2]
    msgs = [e["message"] for e in state["events"]]
    assert any("safety/determinism" in m and "Math.random" in m for m in msgs)
    assert any("TypeScript" in m for m in msgs)
    assert any("does not change over time" in m for m in msgs)
    # the model really was shown the checker's message on the fix attempt
    assert any("Math.random" in p and "Your previous file" in p for p in prompts)

    d = projects.project_dir(pid)
    report = json.loads((d / "output" / "final.validation.json").read_text())
    assert report["ok"] and report["probe"]["video"]["width"] == 1080 and report["probe"]["video"]["frames"] == 270
    stills = run["scenes"]["s01-step-number-1"]["stills"]
    assert stills and all((d / s).is_file() for s in stills)


# ── small context windows ────────────────────────────────────────────────────

def test_parse_context_error_extracts_the_real_numbers():
    msg = "BackendError: Backend error (400): request (602 tokens) exceeds the available context size (512 tokens), try increasing it"
    assert ai.parse_context_error(msg) == (602, 512)
    assert ai.parse_context_error("something else") is None


def test_estimate_is_pessimistic_for_code():
    assert ai.estimate_tokens("x" * 3000) >= 1000


@needs_renderer
def test_tiny_context_stops_with_actionable_guidance_and_never_calls_the_model(isolated, monkeypatch):
    isolated["tokens"] = 512
    pid = projects.create_project("tiny ctx")["id"]
    called = []

    async def model(*a, **k):
        called.append(1)
        return ""

    monkeypatch.setattr(ai, "_ask_model", model)
    state = run_agent(pid)
    run = state["run"]
    assert run["status"] == "needs_attention" and not called
    assert "512-token context window" in run["error"]["message"] and "Model memory" in run["error"]["message"] and run["error"]["code"] == "small_context"
    assert run["context_tokens"] == 512


def test_backend_context_error_becomes_a_plain_message(isolated, monkeypatch):
    pid = projects.create_project("ctx error")["id"]
    monkeypatch.setattr(renderer_jobs, "renderer_status", lambda: {"available": True, "message": None})

    async def model(*a, **k):
        raise RuntimeError("BackendError: Backend error (400): request (602 tokens) exceeds the available context size (512 tokens), try increasing it")

    monkeypatch.setattr(ai, "_ask_model", model)
    state = run_agent(pid)
    run = state["run"]
    assert run["status"] == "needs_attention"
    assert "512-token context window" in run["error"]["message"] and "about 602" in run["error"]["message"]
    assert "BackendError" not in run["error"]["message"]


@needs_renderer
def test_medium_context_uses_compact_prompts_and_still_completes(isolated, monkeypatch):
    isolated["tokens"] = 4096
    pid = projects.create_project("compact")["id"]
    systems = []
    base = scripted_model(plan_text(3), {i: [fenced(EXAMPLE)] for i in (1, 2, 3)}, [])

    async def model(model_id, system, user, max_tokens, temperature=0.4):
        systems.append((system, max_tokens, user))
        return await base(model_id, system, user, max_tokens, temperature)

    monkeypatch.setattr(ai, "_ask_model", model)
    mock_checker(monkeypatch, {f"s0{i}-step-number-{i}": [{"ok": True}] for i in (1, 2, 3)})
    state = run_agent(pid)
    assert state["run"]["status"] == "completed", state["run"]["error"]
    scene_calls = [c for c in systems if "scene" in c[2] and "Plan a video" not in c[2]]
    assert scene_calls and all("HARD RULES" not in c[0] and "Example:" in c[0] for c in scene_calls)  # compact doc, not the full one
    assert all(ai.estimate_tokens(c[0]) + ai.estimate_tokens(c[2]) + c[1] <= 4096 for c in systems)  # every request fits the window
    assert any("compact prompts" in e["message"] for e in state["events"])


def test_prompts_fit_a_4k_window_with_room_for_a_scene_file():
    compact = agent._scene_system(True)
    full = agent._scene_system(False)
    assert ai.estimate_tokens(compact) < 1100  # leaves >2.5k of a 4096 window for the scene brief + code
    assert ai.estimate_tokens(full) > ai.estimate_tokens(compact) * 2
    assert ai.estimate_tokens(agent.PLAN_SYSTEM_COMPACT) < 400


@needs_renderer
def test_repeating_the_same_error_escalates_the_fix_prompt(monkeypatch):
    pid = projects.create_project("repeat")["id"]
    prompts: list[str] = []
    varied = {1: [fenced(EXAMPLE + f"// try {k}\n") for k in range(3)], 2: [fenced(EXAMPLE)], 3: [fenced(EXAMPLE)]}  # different file each time
    monkeypatch.setattr(ai, "_ask_model", scripted_model(plan_text(3), varied, prompts))
    err = {"path": "agent/candidates/x.tsx:3", "message": "TS2305: Module '\"@oprel/kit\"' has no exported member 'Foo'. (Hint: use spring)"}
    fail = {"ok": False, "stage": "types", "issues": [err]}
    mock_checker(monkeypatch, {"s01-step-number-1": [fail, fail, {"ok": True}], "s02-step-number-2": [{"ok": True}], "s03-step-number-3": [{"ok": True}]})
    state = run_agent(pid)
    assert state["run"]["status"] == "completed", state["run"]["error"]
    fixes = [p for p in prompts if "Your previous file" in p and "scene 1 of" in p]
    assert len(fixes) == 2
    assert "same error" not in fixes[0]                      # first failure: normal feedback
    assert "failed with the same error" in fixes[1]          # identical error again: told not to resubmit the same code
    assert "Hint: use spring" in fixes[1]                    # the checker's hint reaches the model


@needs_renderer
def test_identical_resubmission_is_reasked_with_more_variation_instead_of_rechecked(monkeypatch):
    pid = projects.create_project("identical")["id"]
    temps: list[float] = []
    replies = iter([EXAMPLE, EXAMPLE, EXAMPLE + "// different\n"])  # attempt 1, then the same file again, then a changed one

    async def model(model_id, system, user, max_tokens, temperature=0.4):
        if "Plan a video" in user:
            return plan_text(3)
        n = int(re.search(r"scene (\d+) of", user).group(1))
        if n != 1:
            return fenced(EXAMPLE)
        temps.append(temperature)
        return fenced(next(replies))

    monkeypatch.setattr(ai, "_ask_model", model)
    fail = {"ok": False, "stage": "types", "issues": [{"path": "agent/candidates/x.tsx:3", "message": "TS2554: wrong"}]}
    calls = mock_checker(monkeypatch, {"s01-step-number-1": [fail, {"ok": True}], "s02-step-number-2": [{"ok": True}], "s03-step-number-3": [{"ok": True}]})
    state = run_agent(pid)
    assert state["run"]["status"] == "completed", state["run"]["error"]
    assert temps[0] == 0.4 and temps[1] == 0.4 and temps[2] == 0.8  # attempt 2 returned the same file -> re-asked hotter
    assert any("identical file" in e["message"] for e in state["events"])
    assert sum("s01" in c for c in calls) == 2  # the duplicate was never sent to the checker


@needs_renderer
def test_checker_repairs_are_adopted_and_used_as_the_working_copy(monkeypatch):
    pid = projects.create_project("autofix")["id"]
    d = projects.project_dir(pid)
    prompts: list[str] = []
    monkeypatch.setattr(ai, "_ask_model", scripted_model(plan_text(3), {1: [fenced(EXAMPLE), fenced(EXAMPLE + "// v2\n")], 2: [fenced(EXAMPLE)], 3: [fenced(EXAMPLE)]}, prompts))
    real = renderer_jobs.run_worker
    state_ = {"n": 0}

    def fake(op, p, extra=None, timeout=900, cancel=None, holder=None):
        if op != "check-scene":
            return real(op, p, extra, timeout, cancel, holder)
        file = extra[extra.index("--file") + 1]
        if "s01" in file and state_["n"] == 0:
            state_["n"] += 1
            fixed = file.replace(".tsx", ".fixed.tsx")
            (d / fixed).write_text(EXAMPLE + "// repaired by the checker\n", encoding="utf-8")
            return {"event": "result", "ok": False, "stage": "render", "issues": [{"path": "render", "message": "nothing is drawn"}], "warnings": [], "frames": [],
                    "autofixes": [{"rule": "interpolate-arity", "line": 28, "detail": "x"}], "fixedFile": fixed, "exitCode": 0}
        return {"event": "result", "ok": True, "stage": "render", "issues": [], "warnings": [], "frames": [], "exitCode": 0}

    monkeypatch.setattr(renderer_jobs, "run_worker", fake)
    state = run_agent(pid)
    assert state["run"]["status"] == "completed", state["run"]["error"]
    assert any("automatically repaired 1 API slip(s) (interpolate-arity)" in e["message"] for e in state["events"])
    fix_prompt = next(p for p in prompts if "Your previous file" in p and "scene 1 of" in p)
    assert "repaired by the checker" in fix_prompt  # the model is shown the repaired file, so line numbers match


def test_fix_prompt_points_at_the_offending_lines():
    code = "import a\nconst x = 1;\nconst y = interpolate(frame, [0, 1]);\nreturn null;\n"
    issues = [{"path": "agent/candidates/s.tsx:3", "message": "TS2554: Expected 3-4 arguments"}]
    prompt = agent._fix_user("BASE", code, issues, "TypeScript")
    assert "The lines to change" in prompt and "3: const y = interpolate(frame, [0, 1]);" in prompt
    assert "const x = 1;" not in prompt.split("The lines to change")[1]  # only the flagged line, not the whole file
