"""
Model aliases for user-friendly names to official GGUF models.

Oprel Studio Model Registry (2026 Production)
Source: Official & Verified Community GGUFs (Unsloth, Bartowski, Calcuis)
"""

from typing import Dict, List, Optional
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


OFFICIAL_REPOS = {
       "text-generation": {

        # --- QWEN 3 FAMILY (SOTA 2026) ---
        "qwen3-235b": "Qwen/Qwen3-235B-GGUF",     # Flagship Dense
        "qwen3-32b": "Qwen/Qwen3-32B-GGUF",       # Flagship Dense
        "qwen3-14b": "Qwen/Qwen3-14B-GGUF",       # Best All-Rounder
        "qwen3-8b": "Qwen/Qwen3-8B-GGUF",         # Consumer GPU King
        "qwen3-4b": "Qwen/Qwen3-4B-GGUF",         #
        "qwen3-1.7b": "Qwen/Qwen3-1.7B-GGUF",     #
        "qwen3-0.6b": "Qwen/Qwen3-0.6B-GGUF",     # IoT / Embedded

        # --- QWEN 2.5 FAMILY (Reliable Legacy) ---
        "qwen2.5-72b": "Qwen/Qwen2.5-72B-GGUF",     # Flagship Dense
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",

        # --- GEMMA 3 FAMILY (Google) ---        
        "gemma3-1b": "unsloth/gemma-3-1b-it-GGUF", 
        "gemma3-270m": "unsloth/gemma-3-270m-it-GGUF",         #

        # --- GEMMA 2 FAMILY ---
        "gemma2-27b": "lmstudio-community/gemma-2-27b-it-GGUF",
        "gemma2-9b": "lmstudio-community/gemma-2-9b-it-GGUF",
        "gemma2-2b": "lmstudio-community/gemma-2-2b-it-GGUF",

        # --- LLAMA 3.x FAMILY ---
        "llama3.3-70b": "unsloth/Llama-3.3-70B-Instruct-GGUF", # Replaced phantom 8B with real 70B
        "llama3.3-8b": "unsloth/Llama-3.3-8B-Instruct-GGUF",
        "llama3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF",
        "llama3.2-3b": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "llama3.2-1b": "unsloth/Llama-3.2-1B-Instruct-GGUF",
        
        # --- MISTRAL & OTHERS ---
        "mistral-3-3b" : "mistralai/Ministral-3-3B-Reasoning-2512-GGUF",
        "mistral-3-8b" : "mistralai/Ministral-3-8B-Instruct-2512-GGUF",
        "mistral-3-14b" : "mistralai/Ministral-3-14B-Instruct-2512-GGUF",
        "devstral-2-24b": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
        
        # --- Oprel ---
        "oprel-2b": "ragul2607/Oprel-2b-GUFF",

        # --- GPT & OTHERS ---
        "gpt-oss-20b": "unsloth/gpt-oss-20b-GGUF",
        "gpt-oss-120b": "unsloth/gpt-oss-120b-GGUF",
    },

    "coding": {
        # --- QWEN 3 CODER (2026 SOTA) ---
        "qwen3-coder-next-80b" :"Qwen/Qwen3-Coder-Next-GGUF",
        "qwen3-coder-31b":"unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",

        # --- QWEN 2.5 CODER ---
        "qwen2.5-coder-14b": "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "qwen2.5-coder-3b": "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "qwen2.5-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",

        # --- SPECIALISTS ---
        "phi-4-14b": "microsoft/Phi-4-GGUF",
        "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct-GGUF",
        "deepseek-coder-v2-16b": "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "hermes-3-8b": "NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
    },

    "reasoning": {
        "deepseek-r1-14b": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "deepseek-r1-8b": "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "deepseek-r1-7b": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "deepseek-r1-1.5b": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "qwen3-reasoning-7b": "Qwen/Qwen3-7B-Reasoning-GGUF",
        "oprel-2b": "ragul2607/Oprel-2b-GUFF"
    },

    "Text + Vision": {

      # --- QWEN VL (Best OCR/Vision) ---
        "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct-GGUF",
        "qwen3-vl-32b-thinking": "Qwen/Qwen3-VL-32B-Thinking-GGUF",
        "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
        "qwen3-vl-4b-thinking": "Qwen/Qwen3-VL-4B-Thinking-GGUF",
        "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct-GGUF",
        "qwen3-vl-2b-thinking": "Qwen/Qwen3-VL-2B-Thinking-GGUF",
        "qwen2.5-vl-7b": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "qwen2.5-vl-3b": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF",

        # --- Google Gemma 3 VL ---
        "gemma3-vl-27b": "unsloth/gemma-3-27b-it-GGUF",
        "gemma3-vl-12b": "unsloth/gemma-3-12b-it-GGUF",
        "gemma3-vl-4b": "unsloth/gemma-3-4b-it-GGUF",

        # --- Google Gemma 4 
        "gemma 4-vl-5b" : "unsloth/gemma-4-E2B-it-GGUF",
        "gemma 4-vl-8b" : "unsloth/gemma-4-E4B-it-GGUF",
        "gemma 4-vl-26b" : "unsloth/gemma-4-26B-A4B-it-GGUF",
        "gemma 4-vl-31 b" : "unsloth/gemma-4-31B-it-GGUF",
        "gemma 4-vl-5b-uncensored" : "HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive"

    },

    "embeddings": {
        "embeddinggemma-0.3b":"unsloth/embeddinggemma-300m-GGUF",
        "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "bge-m3": "audo/bge-m3-GGUF",
        "snowflake-arctic": "ChristianAzinn/snowflake-arctic-embed-m-gguf",
        "all-minilm-l6-v2": "faldor/all-MiniLM-L6-v2-gguf",
        "e5-small": "intfloat/e5-small-v2-gguf",
    },
    
    # GGUF-only image models for stable-diffusion.cpp backend
    "text-to-image": {
        "ideation" : "HamSFL/Ideation"
    },
}

# Flatten all models into a single dict for backward compatibility
MODEL_ALIASES: Dict[str, str] = {}
for category, models in OFFICIAL_REPOS.items():
    MODEL_ALIASES.update(models)

# Create reverse lookup for repo_id -> alias
REPO_TO_ALIAS: Dict[str, str] = {repo_id: alias for alias, repo_id in MODEL_ALIASES.items()}


def get_best_alias_for_repo(repo_id: str) -> Optional[str]:
    """
    Get the best alias name for a repo ID.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF")
        
    Returns:
        Best alias name (e.g., "deepseek-r1-1.5b") or None if not found
    """
    return REPO_TO_ALIAS.get(repo_id)

# Category metadata with descriptions and icons
CATEGORY_INFO = {
    "text-generation": {
        "name": "Text Generation",
        "icon": "📝",
        "description": "General-purpose chat and instruction-following models (GGUF)",
        "backend": "llama.cpp",
    },
    "coding": {
        "name": "Coding",
        "icon": "👨‍💻",
        "description": "Code generation and software development models (GGUF)",
        "backend": "llama.cpp",
    },
    "reasoning": {
        "name": "Reasoning",
        "icon": "🧠",
        "description": "Advanced reasoning and chain-of-thought models (GGUF)",
        "backend": "llama.cpp",
    },
    "Text + Vision": {
        "name": "Text + Vision",
        "icon": "👁️",
        "description": "Vision-language models for image understanding (GGUF + mmproj)",
        "backend": "llama.cpp",
    },
    "OCR": {
        "name": "OCR",
        "icon": "📄",
        "description": "Optical Character Recognition models (GGUF + mmproj)",
        "backend": "llama.cpp",
    },

    "embeddings": {
        "name": "Embeddings",
        "icon": "📚",
        "description": "Text embeddings for RAG and semantic search (GGUF)",
        "backend": "llama.cpp",
    },
    "text-to-image": {
        "name": "Text-to-Image",
        "icon": "🎨",
        "description": "Image generation from text prompts (GGUF only)",
        "backend": "stable-diffusion.cpp",
    },
}

def resolve_model_id(model_id: str) -> str:
    """
    Resolve user-friendly names to official GGUF model IDs.
    
    Args:
        model_id: User input (e.g., "llama3", "qwencoder")
        
    Returns:
        HuggingFace GGUF model ID
    """
    # Direct match
    if model_id in MODEL_ALIASES:
        resolved = MODEL_ALIASES[model_id]
        logger.info(f"Resolved '{model_id}' -> '{resolved}'")
        return resolved
    
    # Already a GGUF path
    if "/" in model_id and "gguf" in model_id.lower():
        return model_id
    
    # Fuzzy match (case-insensitive)
    model_lower = model_id.lower()
    for alias, gguf_id in MODEL_ALIASES.items():
        if model_lower == alias.lower():
            logger.info(f"Resolved '{model_id}' -> '{gguf_id}'")
            return gguf_id
            
    # Reverse lookup: Check if input matches the filename of any alias target
    # e.g. "Qwen2.5-VL-3B-Instruct-GGUF" -> "unsloth/Qwen2.5-VL-3B-Instruct-GGUF"
    if "/" not in model_id:
        for alias, gguf_id in MODEL_ALIASES.items():
            if "/" in gguf_id:
                repo_filename = gguf_id.split("/")[-1]
                if model_lower == repo_filename.lower():
                     logger.info(f"Resolved filename '{model_id}' -> '{gguf_id}'")
                     return gguf_id
    
    # No match - return original
    return model_id


def list_available_aliases() -> Dict[str, str]:
    """Get all available model aliases."""
    return MODEL_ALIASES.copy()


def list_models_by_category(category: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """
    Get models organized by category.
    
    Args:
        category: Optional category filter (e.g., "coding", "vision")
        
    Returns:
        Dictionary of {category: {alias: repo_id}} or single category if filtered
    """
    if category:
        # Validate category
        if category not in OFFICIAL_REPOS:
            valid_cats = ", ".join(OFFICIAL_REPOS.keys())
            raise ValueError(f"Invalid category '{category}'. Valid: {valid_cats}")
        models = OFFICIAL_REPOS[category]
        return {category: models}
    
    return OFFICIAL_REPOS.copy()


def get_model_category(alias: str) -> Optional[str]:
    """
    Find which category a model alias belongs to.
    
    Args:
        alias: Model alias
        
    Returns:
        Category name or None if not found
    """
    for category, models in OFFICIAL_REPOS.items():
        if alias in models:
            return category

    # Allow repo_id lookups too, so cached model scans can classify
    # models like "HamSFL/Ideation" even when they are not stored by alias.
    for category, models in OFFICIAL_REPOS.items():
        if alias in models.values():
            return category

    return None


def get_categories() -> List[str]:
    """Get list of all available categories."""
    return list(OFFICIAL_REPOS.keys())


def get_category_info(category: str) -> Dict[str, str]:
    """Get metadata about a category."""
    return CATEGORY_INFO.get(category, {
        "name": category,
        "icon": "📦",
        "description": f"{category} models",
        "backend": "unknown"
    })


def get_model_backend(alias: str) -> str:
    """
    Get the backend engine required for a model.
    
    Args:
        alias: Model alias
        
    Returns:
        Backend name for the model category.
    """
    category = get_model_category(alias)
    if not category:
        return "llama.cpp"
    return get_category_info(category).get("backend", "llama.cpp")

def search_aliases(query: str, category: Optional[str] = None) -> List[str]:
    """
    Search for model aliases matching a query.
    
    Args:
        query: Search term
        category: Optional category filter
        
    Returns:
        Sorted list of matching aliases
    """
    query_lower = query.lower()
    
    # Get models to search
    if category:
        if category not in OFFICIAL_REPOS:
            return []
        search_dict = OFFICIAL_REPOS[category]
    else:
        search_dict = MODEL_ALIASES
    
    return sorted([
        alias for alias in search_dict.keys()
        if query_lower in alias.lower()
    ])


