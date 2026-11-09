"""
Curated connector catalog.

Entries describe what a connector can reach, so the UI can show it before install.
Versions are NOT hard-coded: the current version is looked up on the public npm registry
at install time, pinned, and shown to the user.

  network  "none"  the server only touches local resources
           "internet"  it talks to an online service; data can leave this computer
  risk     "write" = exposes tools that change data (asks unless you allow them)
  usage_hint  optional guidance added to the prompt so the model uses the connector when it should
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

CATALOG: List[Dict[str, Any]] = [
    {
        "id": "filesystem",
        "name": "Files",
        "description": "Read and edit files inside folders you choose.",
        "package": "@modelcontextprotocol/server-filesystem",
        "usage_hint": (
            "Use list_directory or search_files to find files before saying a file does not exist, and read_text_file "
            "to read one. Only say a file was changed after write_file or edit_file reported success."
        ),
        "needs_paths": True,  # allowed folders are passed as arguments
        "network": "none",
        "risk": "write",
    },
    {
        "id": "memory",
        "name": "Memory",
        "description": "A small local knowledge graph the assistant can remember things in.",
        "package": "@modelcontextprotocol/server-memory",
        "usage_hint": (
            "Memory holds facts the user asked you to remember. Keep facts about the user on ONE entity named "
            "\"User\". To save a fact: call search_nodes for \"User\" first; if it does not exist call create_entities, "
            "otherwise call add_observations on it. create_entities skips entities that already exist, so read its "
            "result and follow up if something was skipped. When asked about the user or anything they may have told "
            "you, call search_nodes or read_graph BEFORE answering. Never say something was saved or remembered "
            "unless the tool result confirms it."
        ),
        "network": "none",
        "risk": "write",
    },
    {
        "id": "sequential-thinking",
        "name": "Step-by-step thinking",
        "description": "Helps the model break a problem into steps. No data access.",
        "package": "@modelcontextprotocol/server-sequential-thinking",
        "network": "none",
        "risk": "read",
    },
]


def get_catalog_entry(catalog_id: str) -> Optional[Dict[str, Any]]:
    return next((c for c in CATALOG if c["id"] == catalog_id), None)


_PREVIEW_FILE = Path(__file__).with_name("catalog_tools.json")


@lru_cache(maxsize=1)
def _previews() -> Dict[str, Any]:
    try:
        return json.loads(_PREVIEW_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def get_preview(catalog_id: str) -> Dict[str, Any]:
    """
    Tools and prompts a catalog connector provides, so the UI can show them before it is connected.
    This is a snapshot taken from the real server (see "checked_version"); once connected, the live
    list reported by the server replaces it.
    """
    return _previews().get(catalog_id, {"checked_version": None, "tools": [], "prompts": []})
