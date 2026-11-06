from __future__ import annotations

import importlib
import logging
import threading

from oprel.integrations.base import Integration
from oprel.tools.base import Tool

# module path -> attribute holding the Integration instance.
# Adding a new app = new subpackage + one line here.
BUILTIN_INTEGRATIONS: dict[str, str] = {
    "oprel.integrations.azure_devops": "integration",
    "oprel.integrations.scene3d": "integration",
}

_lock = threading.Lock()
_registry: dict[str, Integration] = {}
_loaded = False


def register(integration: Integration) -> None:
    with _lock:
        if integration.id in _registry:
            raise ValueError(f"Integration '{integration.id}' is already registered")
        names = {t.name for t in integration.tools()}
        for other in _registry.values():
            clash = names & {t.name for t in other.tools()}
            if clash:
                raise ValueError(f"Tool name(s) {sorted(clash)} already used by '{other.id}'")
        _registry[integration.id] = integration


def load_builtin_integrations() -> None:
    global _loaded
    if _loaded:
        return
    for module_path, attr in BUILTIN_INTEGRATIONS.items():
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            # A missing optional dependency (e.g. trimesh for scene3d) disables only that integration.
            logging.getLogger(__name__).warning(f"Integration '{module_path}' is unavailable: {exc}")
            continue
        integration = getattr(module, attr)
        if integration.id not in _registry:
            register(integration)
    _loaded = True


def get_integration(integration_id: str) -> Integration | None:
    load_builtin_integrations()
    return _registry.get(integration_id)


def list_integrations() -> list[Integration]:
    load_builtin_integrations()
    return sorted(_registry.values(), key=lambda i: i.name.lower())


def find_tool(tool_name: str) -> tuple[Integration, Tool] | None:
    load_builtin_integrations()
    for integration in _registry.values():
        for tool in integration.tools():
            if tool.name == tool_name:
                return integration, tool
    return None
