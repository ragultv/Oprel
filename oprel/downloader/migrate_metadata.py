"""
Migration script to add metadata for existing cached models
"""

from pathlib import Path
from oprel.core.config import Config
from oprel.downloader.metadata import save_model_metadata, infer_repo_id_from_cache
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


def migrate_existing_models():
    """
    Add metadata for existing cached models that don't have it.
    """
    config = Config()
    cache_dir = config.cache_dir
    
    if not cache_dir.exists():
        logger.info("No cache directory found, nothing to migrate")
        return
    
    migrated_count = 0
    
    try:
        # Find all GGUF files
        for gguf_file in cache_dir.rglob("*.gguf"):
            if not gguf_file.is_file() or gguf_file.stat().st_size == 0:
                continue
                
            filename = gguf_file.name
            
            # Skip mmproj and vision files
            filename_lower = filename.lower()
            if 'mmproj' in filename_lower or filename_lower.startswith('vision-') or filename_lower.startswith('clip-'):
                continue
            
            # Check if metadata already exists
            metadata_dir = cache_dir / ".metadata"
            metadata_file = metadata_dir / f"{filename}.json"
            
            if metadata_file.exists():
                continue  # Already has metadata
            
            # Try to infer repo_id from cache structure
            repo_id = infer_repo_id_from_cache(cache_dir, filename)
            
            if repo_id:
                # Try to detect quantization from filename
                quant = "Unknown"
                name_upper = filename.upper()
                for q in ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16"]:
                    if q in name_upper:
                        quant = q
                        break
                
                # Save metadata
                save_model_metadata(cache_dir, repo_id, quant, gguf_file)
                migrated_count += 1
                logger.info(f"Migrated metadata for {filename} -> {repo_id}")
            else:
                logger.warning(f"Could not infer repo_id for {filename}")
    
    except Exception as e:
        logger.error(f"Error during migration: {e}")
    
    logger.info(f"Migration complete: {migrated_count} models migrated")


if __name__ == "__main__":
    migrate_existing_models()