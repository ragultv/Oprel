"""
Cache management utilities
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def get_storage_root() -> Path:
    """
    Get the active storage root directory for Oprel using 3-tier resolution:
    1. Environment variables: OPREL_CACHE_DIR or OPREL_HOME
    2. Persistent user config: ~/.cache/oprel_storage.json or ~/.oprel/config.json
    3. Default fallback: D:/oprel if exists, else ~/.cache/oprel
    """
    # 1. Environment variable
    env_dir = os.environ.get("OPREL_CACHE_DIR") or os.environ.get("OPREL_HOME")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 2. Config file
    for candidate in [
        Path.home() / ".cache" / "oprel_storage.json",
        Path.home() / ".oprel" / "config.json",
    ]:
        try:
            if candidate.exists():
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    stored = data.get("storage_path")
                    if stored:
                        p = Path(stored).expanduser().resolve()
                        p.mkdir(parents=True, exist_ok=True)
                        return p
        except Exception:
            pass

    # 3. Default fallback: D:/oprel if available, else ~/.cache/oprel
    default_d = Path("D:/oprel")
    try:
        if default_d.exists():
            return default_d.resolve()
    except Exception:
        pass

    fallback = (Path.home() / ".cache" / "oprel").resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def set_storage_root(new_path: str | Path, migrate: bool = False) -> Path:
    """
    Update the active storage root directory.

    Args:
        new_path: New directory path chosen by user
        migrate: Whether to copy existing models/data to new directory

    Returns:
        Resolved Path to new storage directory
    """
    target = Path(new_path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    old_root = None
    try:
        old_root = get_storage_root()
    except Exception:
        pass

    if migrate and old_root and old_root != target and old_root.exists():
        for item in old_root.iterdir():
            dest = target / item.name
            try:
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
            except Exception as e:
                logger.warning(f"Error copying {item} during migration: {e}")

    # Save to persistent config files
    cfg_data = {"storage_path": str(target)}
    for candidate in [
        Path.home() / ".cache" / "oprel_storage.json",
        Path.home() / ".oprel" / "config.json",
    ]:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            with open(candidate, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not write storage config to {candidate}: {e}")

    return target


def get_storage_info() -> Dict[str, Any]:
    """
    Get storage path and drive capacity/free space information.
    """
    root = get_storage_root()
    free_gb = 0.0
    total_gb = 0.0
    try:
        usage = shutil.disk_usage(root)
        free_gb = round(usage.free / (1024 ** 3), 1)
        total_gb = round(usage.total / (1024 ** 3), 1)
    except Exception:
        pass

    return {
        "storage_path": str(root),
        "free_space_gb": free_gb,
        "total_space_gb": total_gb,
    }


def get_cache_path() -> Path:
    """
    Get the cache directory for models.

    Returns:
        Path to model cache directory (<storage_root>/models)
    """
    root = get_storage_root()
    cache_dir = root if root.name == "models" else root / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_binary_cache_path() -> Path:
    """
    Get the cache directory for backend binaries.

    Returns:
        Path to binary cache directory (<storage_root>/bin)
    """
    root = get_storage_root()
    binary_dir = root.parent / "bin" if root.name == "models" else root / "bin"
    binary_dir.mkdir(parents=True, exist_ok=True)
    return binary_dir


def list_cached_models() -> List[dict]:
    """
    List all models currently in cache.

    Returns:
        List of dicts with model info:
        [
            {
                "path": Path,
                "name": str,
                "size_mb": float,
                "modified": datetime
            },
            ...
        ]
    """
    cache_dir = get_cache_path()
    models = []

    from oprel.downloader.metadata import load_model_metadata

    # Find all .gguf files recursively
    for model_file in cache_dir.rglob("*.gguf"):
        try:
            # Only models downloaded via oprel have metadata
            if not load_model_metadata(cache_dir, model_file.name):
                continue

            stat = model_file.stat()
            models.append(
                {
                    "path": model_file,
                    "name": model_file.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                }
            )
        except Exception as e:
            logger.warning(f"Error reading {model_file}: {e}")

    # Sort by size (largest first)
    models.sort(key=lambda x: x["size_mb"], reverse=True)
    return models


def get_cache_size() -> float:
    """
    Calculate total size of cache directory.

    Returns:
        Total size in MB
    """
    cache_dir = get_cache_path()
    total_bytes = 0

    for file_path in cache_dir.rglob("*"):
        if file_path.is_file():
            try:
                total_bytes += file_path.stat().st_size
            except Exception:
                pass

    return round(total_bytes / (1024 * 1024), 2)


def clear_cache(confirm: bool = False) -> int:
    """
    Delete all cached models and binaries.

    Args:
        confirm: If False, raises error (safety check)

    Returns:
        Number of files deleted

    Raises:
        RuntimeError: If confirm=False
    """
    if not confirm:
        raise RuntimeError(
            "Cache clearing requires explicit confirmation. " "Call with confirm=True to proceed."
        )

    cache_root = get_storage_root()

    if not cache_root.exists():
        logger.info("Cache directory does not exist")
        return 0

    # Count files before deletion
    file_count = sum(1 for _ in cache_root.rglob("*") if _.is_file())

    # Delete entire cache directory
    shutil.rmtree(cache_root)
    logger.info(f"Deleted cache directory: {cache_root}")
    logger.info(f"Removed {file_count} files")

    return file_count


def delete_model(model_name: str) -> bool:
    """
    Delete a specific model from cache.

    Args:
        model_name: Name of model file (e.g., "llama-2-7b.Q4_K_M.gguf")

    Returns:
        True if deleted, False if not found
    """
    cache_dir = get_cache_path()

    # Search for model file
    for model_file in cache_dir.rglob(model_name):
        try:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            model_file.unlink()
            logger.info(f"Deleted {model_name} ({size_mb:.1f}MB)")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {model_file}: {e}")
            return False

    logger.warning(f"Model not found in cache: {model_name}")
    return False


def prune_old_models(keep_newest: int = 3) -> int:
    """
    Delete old models, keeping only the N most recently used.

    Args:
        keep_newest: Number of models to keep (default: 3)

    Returns:
        Number of models deleted
    """
    models = list_cached_models()

    if len(models) <= keep_newest:
        logger.info(f"Cache has {len(models)} models, no pruning needed")
        return 0

    # Sort by last modified (most recent first)
    models.sort(key=lambda x: x["modified"], reverse=True)

    # Delete older models
    deleted = 0
    for model in models[keep_newest:]:
        try:
            model["path"].unlink()
            logger.info(f"Pruned old model: {model['name']} ({model['size_mb']}MB)")
            deleted += 1
        except Exception as e:
            logger.error(f"Failed to prune {model['name']}: {e}")

    return deleted


def ensure_cache_space(required_mb: float, max_cache_mb: float = 50000) -> bool:
    """
    Ensure cache has enough space by deleting old models if needed.

    Args:
        required_mb: Space needed in MB
        max_cache_mb: Maximum allowed cache size in MB (default: 50GB)

    Returns:
        True if space is available, False if can't free enough space
    """
    current_size = get_cache_size()

    if current_size + required_mb <= max_cache_mb:
        return True

    logger.warning(
        f"Cache size ({current_size:.1f}MB) + new model ({required_mb:.1f}MB) "
        f"would exceed limit ({max_cache_mb:.1f}MB)"
    )

    # Try to free space by deleting old models
    models = list_cached_models()
    models.sort(key=lambda x: x["modified"])  # Oldest first

    freed = 0.0
    for model in models:
        if current_size - freed + required_mb <= max_cache_mb:
            break

        try:
            model["path"].unlink()
            freed += model["size_mb"]
            logger.info(f"Deleted old model to free space: {model['name']}")
        except Exception as e:
            logger.error(f"Failed to delete {model['name']}: {e}")

    new_size = current_size - freed
    return new_size + required_mb <= max_cache_mb
