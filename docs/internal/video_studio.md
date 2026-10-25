# Video Studio (internal)

Oprel can produce a finished YouTube Short from a one-paragraph idea. A model plans the video scene by scene, writes the motion-graphics
code for each scene itself, renders and checks every scene, fixes what breaks and re-runs, then renders the final MP4. There are no
templates and no built-in scenes: every video is written from scratch by the model you pick.

Status: working end to end. Not yet done: Phase 6 hardening (see "Known limitations").

## How it works

```
idea --> PLAN --> VOICE-OVER --> SCENES (per scene: write -> check -> fix -> re-check) --> ASSEMBLE --> RENDER --> validated MP4
          |                              ^
   (optional review)            the model sees the checker's exact errors
```

1. **Plan** (`oprel/video/agent.py`, `PLAN_SYSTEM`): the chosen text model (any Oprel chat model: local GGUF or a configured provider) writes a
   plain-text plan: title, logline and, per scene, narration, a concrete visual description and a rough length. Plain text rather than JSON on purpose:
   a 1B local model failed 3/3 attempts at JSON. `parse_plan` rejects junk (too-short narration, "10 seconds" timings, vague visuals, wrong total
   length) and the problems go back to the model for up to 3 revisions. Optional: pause for the user to edit the plan.
2. **Voice-over** (optional): each scene's narration goes through Audio Studio TTS (`ai.narrate_project`, Kokoro/Fish). Scene lengths are then
   fitted to the real audio and captions are created per sentence. Scene boundaries are exact; sentence times inside a scene are *estimated*
   from text length because Oprel's Whisper returns text only (`captionsSource: "estimated-from-audio-duration"`). No word-level timestamps are
   ever produced.
3. **Scenes**: for every scene the model writes one React component (`scenes/<id>.tsx`) using `remotion`, `@oprel/kit` (design tokens, backdrop,
   header, easing helpers), optionally `three`. The prompt is `renderer/prompts/scene_api.md` plus a working example (`scene_example.tsx`, which
   is itself covered by tests so the prompt cannot drift from what compiles). Each attempt goes through `check-scene` (below). On failure the model
   gets its previous file and the checker's messages and tries again (default 4 attempts per scene). A scene that exhausts its attempts is marked
   *needs attention*; the other scenes continue. The user can retry it with extra direction, edit the plan, or resume.
4. **Assemble**: the storyboard is built (`type: "custom"` scenes) and validated by the same Zod schema + sandbox policy the renderer uses.
5. **Render**: Remotion renders 1080x1920 H.264/AAC; the output is validated with ffprobe and a full decode before it is moved into place.

Runs are long-lived, persisted (`<project>/agent/run.json`, `events.jsonl`) and resumable. A daemon restart marks the run *interrupted*; Resume continues
from the first unfinished scene. Default budget 4 h (configurable 1-48 h). Stop kills the worker process.

## Context windows (small local models)

The agent asks the llama.cpp server for the model's real `n_ctx` (`ai.model_context_tokens`) at the start of every run segment and adapts:
12k+ tokens (or cloud/unknown): full prompts with a worked example. Below that: compact prompts (`scene_api_compact.md`, `PLAN_SYSTEM_COMPACT`,
the old file is dropped from fix prompts if it would not fit) and every request's `max_tokens` is clamped to what is left of the window.
Below ~3.5k tokens it stops with an explanation instead of failing mid-scene; a backend "exceeds the available context size" error is translated
into the same message. Fix on the runtime side (`runtime/backends/llama_cpp.py`, `vram_context_cap`): a model larger than free VRAM used to be
launched with a 512-token context (the `recommend_max_context` floor) even though layers are offloaded to RAM; it now keeps the configured size
(tier default 4096). Models must be reloaded for this to take effect.

## The scene checker (`check-scene`)

`renderer/src/node/scenecheck.ts`, driven by `bin/worker.ts check-scene`. Stages, each producing messages written for a model to act on:

| Stage | What it catches |
|---|---|
| **static** (`node/sandbox.ts`, TypeScript AST) | imports outside an allow-list, network/eval/DOM/storage/worker access, prototype tricks, external URLs, `<script>`/`<iframe>`/media, CSS `transition`/`animation`, and non-determinism (`Math.random`, `Date`, `performance`, timers, `useFrame`) |
| **types** (`node/stage.ts`) | TypeScript errors with line numbers, using the renderer's compiler options |
| **render** | bundles the scene and renders 5 real frames in headless Chrome (0.5x). Errors: render threw / console errors, nothing drawn, **scene does not change over time**, **content in the bottom 480px** (reserved for captions/app UI). Warnings: content touching the frame edge, slow frames |

Stills are kept per attempt and shown in the UI.

**Small-model resilience.** A 4B local model often ignores feedback (it resent byte-identical files on attempts 2-4). So the pipeline does part of the fixing itself:
`node/autofix.ts` repairs the mechanical API slips the compiler pinpoints (`interpolate` missing its output range, `spring(...)(n)` called like a function,
`spring(frame, fps)` positional args, `random(seed, a, b)`, `.value` on numbers, unknown props on kit components, SVG `textAlign`) and the file is type-checked again
(up to 3 rounds); the repaired copy (`*.fixed.tsx`) becomes the working file. If the model resends an identical file the agent re-asks (same attempt) at a higher
temperature instead of re-checking it, fix prompts show only the offending lines, and a repeated identical error says so. `noImplicitAny` is off for scene type-checking
(it never changes the render). The compact prompt (small contexts) steers away from three.js.

## Security model (AI-written code)

The model's code is data until it passes all of this, and the user must opt in per run (`allow_code`; the API refuses otherwise).

- **Static policy** runs at check time *and again at render time* and in `validate` (so hand-edited files are covered too). It is not the only barrier:
- **Network isolation**: the render browser's page carries a Content-Security-Policy (`render.ts`, `CSP`): `connect-src 'self'`, no frames/objects,
  `form-action 'none'`. It cannot reach the internet or the Oprel daemon (different port). `renderer/test/csp.test.ts` proves it with a control run
  (without the policy the probe reaches another local port; with it, zero requests).
- Scene files can only import the allow-list, so they cannot pull in other project or system files; project files are copied into a staging
  dir and only declared assets are exposed to the browser.
- No shell is ever invoked; worker/ffmpeg arguments are lists.
- **Known gap**: this is a sandbox for the *browser*, not a hard VM boundary. Code runs inside Chromium's own sandbox, but a Chromium 0-day is out of
  scope. The UI says what is checked and lets the user read every file.

Privacy: a cloud provider model is used only when the request sets `allow_cloud` (the UI shows a consent checkbox naming the provider). Local models
never leave the device. Rendering itself never needs a network (the headless Chrome download on first use is the only network need).

## Architecture map

Oprel is a Python package: FastAPI daemon (`oprel/server/app.py`, routes in `oprel/server/routes/`) plus a Next.js static-export UI
(`oprel/webui-react`, served at `/gui`). No Electron.

| Concern | Where |
|---|---|
| API | `oprel/server/routes/video.py` (`/v1/video/...`), behind `LocalGuardMiddleware` |
| Agent / run state | `oprel/video/agent.py` |
| Shared AI helpers (model call, TTS, narration fit, validation) | `oprel/video/ai.py` (reuses `generate_text`, `speech.py`) |
| Projects on disk | `oprel/video/projects.py` -> `<cache>/video_projects/<id>/` (`OPREL_VIDEO_PROJECTS_DIR` overrides) |
| Render/check worker process | `oprel/video/renderer_jobs.py` spawns `node --import tsx renderer/bin/worker.ts <op>` (list args, NDJSON events, stdin `cancel`) |
| Worker + scenes + schema | `oprel/video/renderer/` (Remotion, Zod storyboard schema v1, timeline compiler, built-in scene components, `@oprel/kit`) |
| UI | `oprel/webui-react/components/VideoView.tsx` + `components/video/*`, client in `services/api.ts` (`VIDEO`) |

Project layout: `project.json`, `storyboard.json`, `scenes/*.tsx` (final scene code), `assets/narration.wav`, `agent/run.json`,
`agent/events.jsonl`, `agent/candidates/` (every attempt), `agent/stills/` (checker frames), `output/final.mp4` + `*.validation.json`, `logs/`.

Model calls are **ephemeral**: `ai._ask_model` sets `GenerateParams.ephemeral`, so neither the local nor the provider path creates a saved
conversation, history entry or long-term memory (previously cloud-provider calls showed up in the normal chat list and polluted memory).

Runtime settings that matter for local models: **context window** and **KV cache precision** (`q4_0` is the smallest llama.cpp supports; vision models
default to `f16` under `auto`). Priority: explicit argument > env (`OPREL_CTX_SIZE`, `OPREL_KV_CACHE_TYPE`) > `~/.cache/oprel/runtime_settings.json` > automatic.
Because `Model()` builds a fresh `Config()` on every load, the file takes effect on the next model load without restarting the daemon. Set it in the UI
(Video Studio -> "Model memory", also shown inside a paused "context too small" run) or via `GET|PUT /v1/system/runtime-settings` (PUT can unload the model
so it reloads immediately). An explicit context is honoured rather than capped by the automatic VRAM estimate; Oprel offloads fewer layers instead.

Env: `OPREL_VIDEO_MAX_AGENTS` (default 1 concurrent agent run), `OPREL_VIDEO_MAX_JOBS` (renders), `OPREL_VIDEO_CONCURRENCY` (browser tabs),
`OPREL_VIDEO_GL`, `OPREL_FFMPEG_DIR`.

## HTTP API

`GET /v1/video/status`, `GET /v1/video/tts-models`, `GET|POST /v1/video/projects`, `GET|DELETE /v1/video/projects/{id}`, `POST .../validate`,
agent: `POST .../agent/start`, `GET .../agent?since=<seq>`, `POST .../agent/cancel|resume|approve`, `PUT .../agent/plan`,
`POST .../agent/scenes/{scene}/retry`, `GET .../files/<stills|scenes|candidates>` (whitelisted paths only),
render: `POST .../render`, `GET /v1/video/jobs/{id}`, `POST /v1/video/jobs/{id}/cancel`, `GET .../output/{final|preview}.mp4`.

## Setup

- Node 20+ on PATH, then `npm install` in `oprel/video/renderer` (not shipped in the wheel).
- ffmpeg/ffprobe come from Remotion's compositor package (minimal build). First render downloads Chrome Headless Shell (~120 MB) once.
- For voice-over install a TTS model on the Audio page. Without one the video has no voice and timing comes from the plan.
- Developer commands (from `oprel/video/renderer`): `npm run video:validate|video:preview|video:render|video:check -- --project <dir>`,
  `npx tsx bin/worker.ts check-scene --project <dir> --file <rel.tsx> --duration <frames>`, `npm test`, `npm run typecheck`.

## Tests

- `tests/unit/test_video_agent.py`: plan parsing, run control, resume/cancel/retry/review, privacy gates, interrupted-run detection, file whitelist, and a
  slow end-to-end test with the **real** checker, browser and final render driven by a scripted model (policy, type and static-scene failures are
  detected and fixed).
- `tests/unit/test_video_ai.py`, `test_video_studio.py`: narration fitting, projects, worker jobs, cancel.
- `renderer/test/`: compiler, sandbox policy (35 cases), frame analysis, CSP isolation.

## Known limitations

- **Output quality depends on the director model.** Writing working motion-graphics code needs a capable model. A 1B local model failed the plan
  and scene steps (it fails cleanly with actionable errors). The pipeline was verified end to end with a *scripted stand-in* that returns real plans and
  real scene code (including deliberately broken first attempts); a real strong model has not been run against it yet, so prompts will need tuning.
- The checker cannot judge aesthetics (no vision model is assumed): it verifies the scene renders, animates, draws something and respects the safe zones.
- Caption timing inside a scene is estimated (no aligner installed). No music/SFX mixing yet.
- The renderer's `node_modules` and `.render-cache` live inside the package directory; for a pip-installed package that directory may not be writable.
  Packaging the renderer into a user-data dir is part of hardening.
- A resumed run gets a fresh time budget. Agent runs are serialized (one at a time) by default.
- Output pixel format is requested as yuv420p; ffprobe may report yuvj420p (full-range tag), which plays everywhere we tried.
