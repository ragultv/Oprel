from __future__ import annotations

import sys
from pathlib import Path

from fastapi import Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def get_webui_dir() -> str | None:
    react_ui_paths = [
        Path(__file__).resolve().parents[2] / "webui-react" / "out",
        Path(sys.prefix) / "oprel" / "webui-react" / "out",
    ]

    for path in react_ui_paths:
        if path.exists() and (path / "index.html").exists():
            return str(path)

    try:
        from importlib.resources import files

        path = files("oprel") / "webui-react" / "out"
        if path.joinpath("index.html").exists():
            return str(path)
    except (ImportError, Exception):
        pass

    legacy_ui_paths = [
        Path(sys.prefix) / "oprel" / "webui",
        Path(__file__).parent.parent / "webui",
    ]

    for path in legacy_ui_paths:
        if path.exists():
            return str(path)

    return None


class UIStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope) -> Response:
        # 1. If exact file exists on disk, serve it via super() with proper MIME type
        full_path = Path(self.directory) / path
        if full_path.is_file():
            response = await super().get_response(path, scope)
            if not path or path.endswith(".html") or "." not in Path(path).name:
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

        # 2. Next.js 15+ App Router static export RSC fallback
        if ".__PAGE__.txt" in path:
            alt_path = path.replace(".__PAGE__.txt", "/__PAGE__.txt")
            if (Path(self.directory) / alt_path).is_file():
                return await super().get_response(alt_path, scope)

        # 3. Only return chat/index.html if path is clean route (no file extension)
        if path == "chat" or (path.startswith("chat/") and "." not in Path(path).name):
            chat_index = Path(self.directory) / "chat" / "index.html"
            if chat_index.is_file():
                response = FileResponse(chat_index)
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                return response

        response = await super().get_response(path, scope)
        last_part = path.split("/")[-1] if path else ""
        if not path or path.endswith(".html") or "." not in last_part:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
