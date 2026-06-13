"""
Oprel Daemon Server entrypoint.
"""

from __future__ import annotations

from oprel.server.app import app, run_server

__all__ = ["app", "run_server"]


if __name__ == "__main__":
    run_server()
