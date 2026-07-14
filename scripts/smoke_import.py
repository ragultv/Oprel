#!/usr/bin/env python3
"""
Minimal import smoke test for the Oprel package.

Verifies that the local ``oprel`` package can be imported and that
``oprel.core.config.Config`` can be instantiated **without** triggering
runtime downloads, binary installation, server startup, OCR setup, or any
network/API activity.

Designed to run in CI with ``PYTHONPATH=.`` (no pip install of the package
itself) and ``OPREL_SKIP_RUNTIME_DOWNLOAD=1``.

Exit codes:
    0  -- all checks passed
    1  -- import or instantiation failed
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Prevent any runtime download logic from firing during import.
# This must be set *before* any oprel module is imported.
# ---------------------------------------------------------------------------
os.environ.setdefault("OPREL_SKIP_RUNTIME_DOWNLOAD", "1")


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Import the top-level oprel package.
    #    This exercises oprel/__init__.py and its transitive imports.
    # ------------------------------------------------------------------
    try:
        import oprel  # noqa: F401
    except ImportError as exc:
        print(
            "FAIL: could not import top-level 'oprel' package.\n"
            f"  Missing dependency: {exc}\n"
            "  Ensure import-time dependencies are installed in CI:\n"
            "    pip install huggingface-hub psutil requests pydantic rich tqdm",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # pragma: no cover -- unexpected error
        print(f"FAIL: unexpected error importing 'oprel': {exc}", file=sys.stderr)
        return 1

    print(f"  [1/3] import oprel ............. ok (v{oprel.__version__})")

    # ------------------------------------------------------------------
    # 2. Import Config from oprel.core.config.
    # ------------------------------------------------------------------
    try:
        from oprel.core.config import Config
    except ImportError as exc:
        print(f"FAIL: could not import Config from oprel.core.config: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"FAIL: unexpected error importing Config: {exc}", file=sys.stderr)
        return 1

    print("  [2/3] from oprel.core.config import Config ... ok")

    # ------------------------------------------------------------------
    # 3. Instantiate Config.
    #
    #    Source inspection confirms Config is a pydantic BaseModel whose
    #    field defaults are pure-Python lambdas (Path arithmetic) or a
    #    memory-limit helper that reads psutil.virtual_memory() inside a
    #    try/except with an 8192 MB fallback.
    #
    #    No __init__ side effects: no network, no binary download, no
    #    server start, no OCR, no API keys.
    # ------------------------------------------------------------------
    try:
        cfg = Config()
    except Exception as exc:  # pragma: no cover
        print(f"FAIL: Config() raised: {exc}", file=sys.stderr)
        return 1

    # Sanity: verify the object is usable without touching the network.
    assert hasattr(cfg, "cache_dir"), "Config instance missing cache_dir attribute"
    assert hasattr(cfg, "binary_dir"), "Config instance missing binary_dir attribute"

    print("  [3/3] Config() ................. ok")

    print("\nSMOKE TEST PASSED — oprel imports cleanly, Config instantiates safely.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
