"""
ComfyUI Backend Integration for Image and Video Generation.

This backend integrates with ComfyUI (https://github.com/comfyanonymous/ComfyUI)
for production-grade Stable Diffusion and video generation workflows.

ComfyUI must be running separately on the system.
Default: http://127.0.0.1:8188
"""

import json
import time
import uuid
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path

from oprel.utils.logging import get_logger

logger = get_logger(__name__)


class ComfyUIClient:
    """
    Client for ComfyUI API integration.
    
    Handles image and video generation through ComfyUI's workflow system.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        """
        Initialize ComfyUI client.
        
        Args:
            base_url: ComfyUI server URL
        """
        self.base_url = base_url.rstrip('/')
        self.client_id = str(uuid.uuid4())
        
    def is_available(self) -> bool:
        """Check if ComfyUI server is running."""
        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_models(self, model_type: str = "checkpoints") -> List[str]:
        """
        Get available models from ComfyUI.
        
        Args:
            model_type: Type of models (checkpoints, loras, vaes, etc.)
            
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/object_info")
            if response.status_code == 200:
                data = response.json()
                # ComfyUI stores model info in object_info
                if model_type == "checkpoints":
                    return data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
                return []
        except Exception as e:
            logger.error(f"Failed to get ComfyUI models: {e}")
            return []
    
    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """
        Queue a workflow for execution.
        
        Args:
            workflow: ComfyUI workflow JSON
            
        Returns:
            Prompt ID for tracking
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        response = requests.post(
            f"{self.base_url}/prompt",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        return result["prompt_id"]
    
    def get_history(self, prompt_id: str) -> Optional[Dict]:
        """Get execution history for a prompt."""
        try:
            response = requests.get(f"{self.base_url}/history/{prompt_id}")
            if response.status_code == 200:
                data = response.json()
                return data.get(prompt_id)
        except Exception as e:
            logger.debug(f"Failed to get history: {e}")
        return None
    
    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download generated image from ComfyUI."""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }
        
        response = requests.get(
            f"{self.base_url}/view",
            params=params
        )
        response.raise_for_status()
        return response.content
    
    def wait_for_completion(self, prompt_id: str, timeout: int = None) -> Dict:
        """
        Wait for workflow to complete.
        
        Args:
            prompt_id: Prompt ID to wait for
            timeout: Maximum wait time in seconds (None = infinite)
            
        Returns:
            Execution history
        """
        start_time = time.time()
        
        while True:
            history = self.get_history(prompt_id)
            
            if history and "outputs" in history:
                # Workflow completed
                return history
            
            # Check timeout only if set
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Workflow {prompt_id} did not complete within {timeout}s")
            
            time.sleep(1)



class ComfyUIImageGenerator:
    """
    High-level interface for image generation via ComfyUI.
    """
    
    def __init__(self, client: ComfyUIClient):
        self.client = client
    
    def generate_txt2img(
        self,
        prompt: str,
        checkpoint: str,  # REQUIRED: User must specify model
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seed: int = -1,
        timeout: int = None  # No timeout by default - wait indefinitely
    ) -> bytes:
        """
        Generate image from text using ComfyUI.
        
        Args:
            prompt: Text prompt
            checkpoint: Model checkpoint name (REQUIRED - must specify model)
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            steps: Sampling steps
            cfg_scale: CFG scale
            sampler: Sampler name
            scheduler: Scheduler name
            seed: Random seed (-1 for random)
            timeout: Maximum wait time in seconds (None = no limit)
            
        Returns:
            Image bytes (PNG)
            
        Raises:
            ValueError: If checkpoint is not specified or not found
            RuntimeError: If generation fails
        """
        if seed == -1:
            seed = int(time.time())
        
        # Validate checkpoint is provided
        if not checkpoint:
            raise ValueError(
                "Model checkpoint is required for image generation.\n"
                "Please specify a model name, e.g.:\n"
                "  generator.generate_txt2img(prompt='...', checkpoint='flux-1-schnell.safetensors')\n"
                "  oprel gen-image flux-1-schnell 'your prompt'"
            )
        
        # Get available checkpoints
        checkpoints = self.client.get_models("checkpoints")
        if not checkpoints:
            raise RuntimeError(
                "No checkpoints available in ComfyUI.\n"
                "Please download models first:\n"
                "  oprel setup image"
            )
        
        # Validate checkpoint exists
        if checkpoint not in checkpoints:
            available = '\n  - '.join(checkpoints[:5])  # Show first 5
            raise ValueError(
                f"Model '{checkpoint}' not found in ComfyUI.\n\n"
                f"Available models:\n  - {available}\n\n"
                f"Run 'oprel gen-image --help' to see model aliases."
            )
        
        model_name = checkpoint
        logger.info(f"Using model: {model_name}")
        
        # Detect model family
        is_turbo = "turbo" in model_name.lower()
        is_sdxl = "xl" in model_name.lower() or "sdxl" in model_name.lower() or "flux" in model_name.lower()
        is_sd15 = not is_sdxl
        
        # FIX 2: Force turbo-safe parameters (NON-NEGOTIABLE)
        if is_turbo:
            logger.info("🔥 Turbo model detected - forcing turbo-safe parameters")
            steps = min(steps, 4)  # Max 4 steps
            cfg_scale = 1.0  # MUST be 1.0
            sampler = "euler"  # Only euler works reliably
            scheduler = "normal"  # Only normal works
            logger.info(f"  → Forced: steps={steps}, cfg={cfg_scale}, sampler={sampler}")
        
        # Ensure resolution is divisible by 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        # Build workflow based on model family
        if is_sdxl:
            workflow = self._build_sdxl_workflow(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg_scale,
                sampler=sampler,
                scheduler=scheduler,
                seed=seed,
                checkpoint=model_name
            )
        else:
            workflow = self._build_sd15_workflow(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg_scale,
                sampler=sampler,
                scheduler=scheduler,
                seed=seed,
                checkpoint=model_name
            )
        
        # Queue and wait
        prompt_id = self.client.queue_prompt(workflow)
        logger.info(f"Queued generation: {prompt_id}")
        
        history = self.client.wait_for_completion(prompt_id, timeout=timeout)
        
        # Extract image
        outputs = history.get("outputs", {})
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                image_info = node_output["images"][0]
                return self.client.get_image(
                    filename=image_info["filename"],
                    subfolder=image_info.get("subfolder", ""),
                    folder_type=image_info.get("type", "output")
                )
        
        raise RuntimeError("No image generated")
    
    def _build_sd15_workflow(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        sampler: str,
        scheduler: str,
        seed: int,
        checkpoint: str
    ) -> Dict:
        """Build SD 1.5 workflow with explicit VAE (FIX 5)."""
        
        logger.info(f"Building SD 1.5 workflow: {checkpoint}, {width}x{height}, steps={steps}, cfg={cfg}")
        
        # FIX 3: Fixed sampler names (removed typo "dpm pp_2m_sde")
        valid_samplers = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", 
                          "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", 
                          "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "ddim", "uni_pc"]
        
        if sampler not in valid_samplers:
            sampler = "euler"
            logger.warning(f"Invalid sampler, using euler")
        
        # SD 1.5 workflow with explicit VAE
        workflow = {
            # Checkpoint Loader
            "4": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"}
            },
            # Positive Prompt
            "6": {
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Positive)"}
            },
            # Negative Prompt
            "7": {
                "inputs": {
                    "text": negative_prompt if negative_prompt else "",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Negative)"}
            },
            # Empty Latent Image
            "5": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Empty Latent Image"}
            },
            # KSampler
            "3": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            # VAE Decode (use checkpoint VAE)
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]  # From checkpoint
                },
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"}
            },
            # Save Image
            "9": {
                "inputs": {
                    "filename_prefix": "oprel",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image"}
            }
        }
        
        return workflow
    
    def _build_sdxl_workflow(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        sampler: str,
        scheduler: str,
        seed: int,
        checkpoint: str
    ) -> Dict:
        """Build SDXL-specific workflow (FIX 4)."""
        
        logger.info(f"Building SDXL workflow: {checkpoint}, {width}x{height}, steps={steps}, cfg={cfg}")
        
        # SDXL uses same samplers as SD 1.5
        valid_samplers = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", 
                          "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", 
                          "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "ddim", "uni_pc"]
        
        if sampler not in valid_samplers:
            sampler = "euler"
            logger.warning(f"Invalid sampler, using euler")
        
        # SDXL workflow - similar to SD 1.5 but SDXL models handle CLIP internally
        workflow = {
            # Checkpoint Loader - works for SDXL
            "4": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load SDXL Checkpoint"}
            },
            # Positive Prompt (SDXL CLIP)
            "6": {
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Positive)"}
            },
            # Negative Prompt (SDXL CLIP)
            "7": {
                "inputs": {
                    "text": negative_prompt if negative_prompt else "",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Negative)"}
            },
            # Empty Latent Image
            "5": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {"title": "Empty Latent Image"}
            },
            # KSampler
            "3": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            # VAE Decode
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]  # SDXL checkpoint includes good VAE
                },
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"}
            },
            # Save Image
            "9": {
                "inputs": {
                    "filename_prefix": "oprel_sdxl",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image"}
            }
        }
        
        return workflow


def check_comfyui_available() -> bool:
    """Check if ComfyUI is available."""
    client = ComfyUIClient()
    return client.is_available()


def get_comfyui_status() -> Dict[str, Any]:
    """
    Get ComfyUI status and configuration.
    
    Returns:
        Status dict with 'available', 'url', 'models'
    """
    client = ComfyUIClient()
    available = client.is_available()
    
    status = {
        "available": available,
        "url": client.base_url,
        "models": []
    }
    
    if available:
        try:
            models = client.get_models("checkpoints")
            status["models"] = models
            status["model_count"] = len(models)
        except Exception as e:
            logger.error(f"Failed to get ComfyUI models: {e}")
    
    return status
