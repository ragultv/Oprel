"""
Model metadata management for tracking original repo IDs
"""

import json
from pathlib import Path
from typing import Optional, Dict
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def save_model_metadata(cache_dir: Path, model_id: str, quantization: str, file_path: Path):
    """
    Save metadata for a downloaded model.
    
    Args:
        cache_dir: Cache directory
        model_id: Original HuggingFace repo ID (e.g., "unsloth/gemma-3-1b-it-GGUF")
        quantization: Quantization level (e.g., "Q4_K_M")
        file_path: Path to the downloaded GGUF file
    """
    try:
        # Create metadata directory if it doesn't exist
        metadata_dir = cache_dir / ".metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Use filename as key
        filename = file_path.name
        metadata_file = metadata_dir / f"{filename}.json"
        
        metadata = {
            "repo_id": model_id,
            "quantization": quantization,
            "filename": filename,
            "file_path": str(file_path),
        }
        
        metadata_file.write_text(json.dumps(metadata, indent=2))
        logger.debug(f"Saved metadata for {filename}: {model_id}")
        
    except Exception as e:
        logger.warning(f"Failed to save metadata for {file_path.name}: {e}")


def load_model_metadata(cache_dir: Path, filename: str) -> Optional[Dict[str, str]]:
    """
    Load metadata for a model file.
    
    Args:
        cache_dir: Cache directory
        filename: GGUF filename (e.g., "gemma-3-1b-it-BF16.gguf")
        
    Returns:
        Metadata dict with repo_id, quantization, etc., or None if not found
    """
    try:
        metadata_dir = cache_dir / ".metadata"
        metadata_file = metadata_dir / f"{filename}.json"
        
        if not metadata_file.exists():
            return None
        
        metadata = json.loads(metadata_file.read_text())
        return metadata
        
    except Exception as e:
        logger.debug(f"Failed to load metadata for {filename}: {e}")
        return None


def get_repo_id_from_filename(cache_dir: Path, filename: str) -> Optional[str]:
    """
    Get the original repo ID from a filename.
    
    Args:
        cache_dir: Cache directory
        filename: GGUF filename
        
    Returns:
        Original repo ID or None
    """
    metadata = load_model_metadata(cache_dir, filename)
    if metadata:
        return metadata.get("repo_id")
    return None


def infer_repo_id_from_cache(cache_dir: Path, filename: str) -> Optional[str]:
    """
    Try to infer the repo ID from the cache directory structure.
    
    HuggingFace cache structure: cache_dir/models--org--name/snapshots/...
    
    Args:
        cache_dir: Cache directory
        filename: GGUF filename
        
    Returns:
        Inferred repo ID or None
    """
    try:
        # Search for the file in cache
        for gguf_file in cache_dir.rglob(filename):
            if gguf_file.name == filename:
                # Try to extract repo ID from path
                # Path format: cache_dir/models--org--name/snapshots/.../file.gguf
                parts = gguf_file.parts
                for i, part in enumerate(parts):
                    if part.startswith("models--"):
                        # Convert models--org--name to org/name
                        repo_parts = part.replace("models--", "").split("--")
                        if len(repo_parts) >= 2:
                            repo_id = "/".join(repo_parts)
                            logger.debug(f"Inferred repo_id for {filename}: {repo_id}")
                            return repo_id
        
        return None
        
    except Exception as e:
        logger.debug(f"Failed to infer repo_id for {filename}: {e}")
        return None
