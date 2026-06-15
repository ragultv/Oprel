from __future__ import annotations

import asyncio
import signal
import sys
import time as time_module
from contextlib import asynccontextmanager
from datetime import datetime

from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from oprel.server.domain.state import get_state
from oprel.server.services.context import CONFIG, logger, kill_orphaned_backends, write_daemon_pid
from oprel.server.services.model_state import cleanup_models, monitor_idle_models, IDLE_TIMEOUT_SECONDS
from oprel.server.services.webui import UIStaticFiles, get_webui_dir
from oprel.server.routes import (
    health,
    metrics,
    knowledge,
    models,
    downloads,
    generation,
    conversations,
    users,
    openai_compat,
    ollama_compat,
    images,
    providers,
    system,
    skills,
    ocr,
)


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def get_status_color(status_code: int) -> str:
    if 200 <= status_code < 300:
        return Colors.GREEN
    if 300 <= status_code < 400:
        return Colors.CYAN
    if 400 <= status_code < 500:
        return Colors.YELLOW
    return Colors.RED


def format_duration(duration_ms: float) -> str:
    if duration_ms < 1:
        return f"{duration_ms * 1000:.2f}µs"
    if duration_ms < 1000:
        return f"{duration_ms:.2f}ms"
    return f"{duration_ms / 1000:.2f}s"


class GinStyleLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time_module.perf_counter()
        response = await call_next(request)
        duration = (time_module.perf_counter() - start_time) * 1000

        status_code = response.status_code
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        timestamp = datetime.now().strftime("%Y/%m/%d - %H:%M:%S")
        status_color = get_status_color(status_code)

        log_line = (
            f"{Colors.GRAY}[OPREL]{Colors.RESET} "
            f"{timestamp} "
            f"{Colors.GRAY}|{Colors.RESET} "
            f"{status_color}{Colors.BOLD}{status_code}{Colors.RESET} "
            f"{Colors.GRAY}|{Colors.RESET} "
            f"{format_duration(duration):>12} "
            f"{Colors.GRAY}|{Colors.RESET} "
            f"{Colors.WHITE}{client_ip:>15}{Colors.RESET} "
            f"{Colors.GRAY}|{Colors.RESET} "
            f"{Colors.BLUE}{method:<8}{Colors.RESET} "
            f"{Colors.MAGENTA}\"{path}\"{Colors.RESET}"
        )
        print(log_line)
        return response


class StripApiPrefixMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.scope["path"].startswith("/api/"):
            stripped = request.scope["path"][4:]
            request.scope["path"] = stripped
            request.scope["raw_path"] = stripped.encode()
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = get_state()

    print(f"{Colors.GREEN}Oprel daemon starting...{Colors.RESET}")
    print(f"Cache Dir: {CONFIG.cache_dir}")
    print(f"Idle model timeout: {IDLE_TIMEOUT_SECONDS / 60:.0f} minutes")

    try:
        kill_orphaned_backends()
    except Exception as exc:
        logger.warning(f"Error cleaning orphaned backends on startup: {exc}")

    try:
        from oprel.downloader.migrate_metadata import migrate_existing_models

        migrate_existing_models()
    except Exception as exc:
        logger.warning(f"Error during metadata migration: {exc}")

    write_daemon_pid()

    state.cleanup_task = asyncio.create_task(monitor_idle_models())
    logger.info("Started idle model monitoring task")

    yield

    print("\nReceived shutdown signal, cleaning up...")
    cleanup_models()

    if state.cleanup_task:
        state.cleanup_task.cancel()


def _signal_handler(signum, frame):
    print("\nReceived shutdown signal, cleaning up...")
    cleanup_models()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

app = FastAPI(lifespan=lifespan, title="Oprel Daemon")
app.add_middleware(GinStyleLoggingMiddleware)
app.add_middleware(StripApiPrefixMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-ID"],
)

webui_dir = get_webui_dir()
if webui_dir:

    @app.get("/gui/chat")
    @app.get("/gui/chat/{path:path}")
    async def chat_ui_fallback(path: str = ""):
        chat_index = Path(webui_dir) / "chat" / "index.html"
        if chat_index.exists():
            return FileResponse(chat_index)
        return Response(status_code=404)

if webui_dir:
    app.mount("/gui", UIStaticFiles(directory=str(webui_dir), html=True), name="gui")

app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(knowledge.router)
app.include_router(models.router)
app.include_router(downloads.router)
app.include_router(generation.router)
app.include_router(conversations.router)
app.include_router(users.router)
app.include_router(openai_compat.router)
app.include_router(ollama_compat.router)
app.include_router(images.router)
app.include_router(providers.router)
app.include_router(system.router)
app.include_router(skills.router)
app.include_router(ocr.router)


def run_server(host: str = "127.0.0.1", port: int = 11435):
    import uvicorn

    coral = "\033[38;2;230;115;90m"
    bold = "\033[1m"
    reset = "\033[0m"

    banner = (
       f"""{coral}{bold}
   ██████╗ ██████╗ ██████╗ ███████╗██╗
  ██╔═══██╗██╔══██╗██╔══██╗██╔════╝██║
  ██║   ██║██████╔╝██████╔╝█████╗  ██║
  ██║   ██║██╔═══╝ ██╔══██╗██╔══╝  ██║
  ╚██████╔╝██║     ██║  ██║███████╗███████╗
   ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝
{reset}"""
    )
    print(banner)
    print(f" {Colors.GRAY}Oprel Daemon v0.6.2{Colors.RESET}\n")
    print(f"  Listening on: {Colors.CYAN}http://{host}:{port}{Colors.RESET}")
    print(f"  Press {Colors.YELLOW}Ctrl+C{Colors.RESET} to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)
