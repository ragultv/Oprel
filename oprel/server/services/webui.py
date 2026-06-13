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
        if path == "chat" or path.startswith("chat/"):
            response = FileResponse(Path(self.directory) / "chat" / "index.html")
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
