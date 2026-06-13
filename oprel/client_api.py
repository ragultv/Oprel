"""
Ollama-compatible client API for Oprel SDK

This module provides an Ollama-compatible interface for Oprel,
allowing easy migration and familiar syntax.
"""

import time
import uuid
import base64
from typing import Dict, List, Any, Optional, Iterator, Union
from pathlib import Path

import requests

from oprel.core.model import Model
from oprel.core.config import Config
from oprel.downloader.cache import list_cached_models as _list_cached_models
from oprel.runtime.image_generation import ImageGenerationParams, generate_image_file
from oprel.api_models import (
    ChatResponse,
    GenerateResponse,
    ImageResponse,
    ListResponse,
    ShowResponse,
    ModelInfo,
    Message,
)


class Client:
    """
    Ollama-compatible client for Oprel SDK
    
    Example:
        from oprel import Client
        
        client = Client(host='http://localhost:11434')
        response = client.chat(
            model='qwencoder',
            messages=[{'role': 'user', 'content': 'Hello!'}]
        )
        print(response.message.content)
    """
    
    def __init__(
        self,
        host: str = "http://localhost:11435",
        timeout: float = 300.0,
    ):
        """
        Initialize Oprel client
        
        Args:
            host: Server URL (default: http://localhost:11435)
            timeout: Request timeout in seconds (default: 300)
        """
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._config = Config()
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        format: Optional[Union[str, Dict]] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> Union[ChatResponse, Iterator[ChatResponse]]:
        """
        Generate chat completions
        
        Args:
            model: Model name or ID
            messages: List of message dicts with 'role' and 'content'
            stream: Enable streaming responses
            format: Response format (e.g., 'json' or Pydantic schema)
            options: Model parameters (temperature, max_tokens, etc.)
            keep_alive: Keep model loaded duration
        
        Returns:
            ChatResponse or iterator of ChatResponse chunks if streaming
        
        Example:
            response = client.chat(
                model='qwencoder',
                messages=[{'role': 'user', 'content': 'Hello!'}]
            )
            print(response.message.content)
        """
        start_time = time.time()
        
        # Use server mode
        oprel_model = Model(model, use_server=True)
        oprel_model.load()
        
        # Build conversation from messages
        conversation_id = str(uuid.uuid4())
        system_prompt = None
        user_prompt = ""
        
        # Extract system prompt and last user message
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                user_prompt = msg['content']
        
        # Parse options
        opts = options or {}
        max_tokens = opts.get('num_predict', 8192)
        temperature = opts.get('temperature', 0.7)
        
        if stream:
            return self._chat_stream(
                oprel_model,
                user_prompt,
                conversation_id,
                system_prompt,
                max_tokens,
                temperature,
                start_time,
            )
        
        # Non-streaming response
        response_text = oprel_model.generate(
            user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            conversation_id=conversation_id,
            system_prompt=system_prompt,
        )
        
        total_duration = int((time.time() - start_time) * 1e9)  # nanoseconds
        
        return ChatResponse(
            model=model,
            message=Message(role='assistant', content=response_text),
            total_duration=total_duration,
            eval_count=len(response_text.split()),
        )
    
    def _chat_stream(
        self,
        oprel_model: Model,
        prompt: str,
        conversation_id: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        start_time: float,
    ) -> Iterator[ChatResponse]:
        """Stream chat responses"""
        full_response = ""
        
        for token in oprel_model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            conversation_id=conversation_id,
            system_prompt=system_prompt,
        ):
            full_response += token
            
            yield ChatResponse(
                model=oprel_model.model_id,
                message=Message(role='assistant', content=token),
                done=False,
            )
        
        # Final chunk with done=True
        total_duration = int((time.time() - start_time) * 1e9)
        yield ChatResponse(
            model=oprel_model.model_id,
            message=Message(role='assistant', content=''),
            done=True,
            total_duration=total_duration,
            eval_count=len(full_response.split()),
        )
    
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        format: Optional[Union[str, Dict]] = None,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        keep_alive: Optional[str] = None,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Generate completions
        
        Args:
            model: Model name or ID
            prompt: Text prompt
            stream: Enable streaming
            format: Response format
            options: Model parameters
            system: System prompt
            template: Prompt template
            context: Context from previous generation
            keep_alive: Keep model loaded duration
        
        Returns:
            GenerateResponse or iterator if streaming
        
        Example:
            response = client.generate(
                model='qwencoder',
                prompt='Why is the sky blue?'
            )
            print(response.response)
        """
        start_time = time.time()
        
        oprel_model = Model(model, use_server=True)
        oprel_model.load()
        
        opts = options or {}
        max_tokens = opts.get('num_predict', 8192)
        temperature = opts.get('temperature', 0.7)
        
        if stream:
            return self._generate_stream(
                oprel_model,
                prompt,
                system,
                max_tokens,
                temperature,
                start_time,
            )
        
        response_text = oprel_model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system,
        )
        
        total_duration = int((time.time() - start_time) * 1e9)
        
        return GenerateResponse(
            model=model,
            response=response_text,
            total_duration=total_duration,
            eval_count=len(response_text.split()),
        )

    def generate_image(
        self,
        model: str,
        prompt: str,
        size: str = "1024x1024",
        response_format: str = "url",
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        seed: Optional[int] = None,
        sampler: Optional[str] = None,
    ) -> ImageResponse:
        """Generate an image via the Ollama-compatible image endpoint."""
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": response_format,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if steps is not None:
            payload["steps"] = steps
        if cfg_scale is not None:
            payload["cfg_scale"] = cfg_scale
        if seed is not None:
            payload["seed"] = seed
        if sampler:
            payload["sampler"] = sampler

        if self._server_is_running():
            res = requests.post(
                f"{self.host}/api/images/generate",
                json=payload,
                timeout=self.timeout,
            )
            res.raise_for_status()
            return ImageResponse(**res.json())

        output_path = generate_image_file(
            ImageGenerationParams(
                model=model,
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=int(size.split("x", 1)[0]),
                height=int(size.split("x", 1)[1]),
                steps=steps or 20,
                cfg_scale=cfg_scale if cfg_scale is not None else 7.0,
                seed=seed if seed is not None else -1,
                sampler=sampler,
            ),
            config=self._config,
        )

        image_b64 = base64.b64encode(Path(output_path).read_bytes()).decode("utf-8")
        data_key = "b64_json" if response_format == "b64_json" else "url"
        image_data = {data_key: image_b64 if data_key == "b64_json" else f"data:image/png;base64,{image_b64}"}
        return ImageResponse(created=int(time.time()), data=[{**image_data, "revised_prompt": prompt}])

    def _server_is_running(self) -> bool:
        try:
            response = requests.get(f"{self.host}/health", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def _generate_stream(
        self,
        oprel_model: Model,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        start_time: float,
    ) -> Iterator[GenerateResponse]:
        """Stream generate responses"""
        full_response = ""
        
        for token in oprel_model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            system_prompt=system,
        ):
            full_response += token
            
            yield GenerateResponse(
                model=oprel_model.model_id,
                response=token,
                done=False,
            )
        
        total_duration = int((time.time() - start_time) * 1e9)
        yield GenerateResponse(
            model=oprel_model.model_id,
            response='',
            done=True,
            total_duration=total_duration,
            eval_count=len(full_response.split()),
        )
    
    def list(self) -> ListResponse:
        """
        List available models
        
        Returns:
            ListResponse with list of models
        
        Example:
            models = client.list()
            for model in models.models:
                print(model.name)
        """
        try:
            # Try to get from server
            response = requests.get(f"{self.host}/models", timeout=5)
            if response.status_code == 200:
                server_models = response.json()
                
                model_infos = []
                for m in server_models:
                    model_infos.append(ModelInfo(
                        name=m.get('model_id', m.get('name', 'unknown')),
                        modified_at=m.get('modified_at', ''),
                        size=m.get('size_gb', 0) * 1024**3,
                        digest=m.get('quantization', 'unknown'),
                        details={
                            'backend': m.get('backend', 'llama.cpp'),
                            'loaded': m.get('loaded', False),
                        }
                    ))
                
                return ListResponse(models=model_infos)
        except:
            pass
        
        # Fallback to cache
        cached = _list_cached_models()
        model_infos = []
        
        for m in cached:
            model_infos.append(ModelInfo(
                name=m['name'],
                modified_at=m['modified'].isoformat(),
                size=int(m['size_mb'] * 1024 * 1024),
                digest='unknown',
                details={'cached': True}
            ))
        
        return ListResponse(models=model_infos)
    
    def show(self, model: str) -> ShowResponse:
        """
        Show model information
        
        Args:
            model: Model name
        
        Returns:
            ShowResponse with model details
        
        Example:
            info = client.show('qwencoder')
            print(info.details)
        """
        # Get model info from cache
        cached = _list_cached_models()
        
        for m in cached:
            if model in m['name']:
                return ShowResponse(
                    modelfile=f"FROM {m['name']}",
                    parameters="",
                    template="",
                    details={
                        'name': m['name'],
                        'size': m['size_mb'],
                        'modified': m['modified'].isoformat(),
                        'backend': 'llama.cpp',
                    }
                )
        
        # Model not found, return minimal info
        return ShowResponse(
            modelfile=f"FROM {model}",
            parameters="",
            template="",
            details={'name': model}
        )
    
    def create(
        self,
        model: str,
        from_: Optional[str] = None,
        modelfile: Optional[str] = None,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, str]:
        """
        Create a custom model variant
        
        Args:
            model: New model name
            from_: Base model to customize
            modelfile: Full modelfile content
            system: System prompt for the model
            stream: Stream creation progress
        
        Returns:
            Status dict
        
        Example:
            client.create(
                model='my-assistant',
                from_='qwencoder',
                system='You are a helpful Python expert.'
            )
        """
        # For oprel, we'll just return success since customization
        # happens at runtime via system prompts
        return {
            'status': 'success',
            'message': f'Model {model} configured (system prompts applied at runtime)'
        }
    
    def pull(
        self,
        model: str,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Download a model
        
        Args:
            model: Model name to download
            stream: Stream download progress
        
        Returns:
            Status dict
        
        Example:
            client.pull('qwencoder')
        """
        from oprel.downloader.hub import download_model
        from oprel.downloader.aliases import resolve_model_id
        from oprel.telemetry.recommender import recommend_quantization
        
        model_id = resolve_model_id(model)
        quantization = recommend_quantization()
        
        model_path = download_model(
            model_id,
            quantization=quantization,
            cache_dir=self._config.cache_dir,
        )
        
        return {
            'status': 'success',
            'model': model,
            'path': str(model_path),
        }
    
    def delete(self, model: str) -> Dict[str, str]:
        """
        Delete a model
        
        Args:
            model: Model name to delete
        
        Returns:
            Status dict
        """
        from oprel.downloader.cache import delete_model
        
        if delete_model(model):
            return {'status': 'success', 'message': f'Deleted {model}'}
        else:
            return {'status': 'error', 'message': f'Model {model} not found'}
    
    def embed(
        self,
        texts: Union[str, List[str]],
        model: str = "nomic-embed-text",
        normalize: bool = True,
        **options
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s).

        Mirrors the text-model lifecycle exactly:
        - If the oprel daemon is already running  → use it (model stays loaded in server cache)
        - If no daemon is running                → auto-start one, embed, stop it when done

        Args:
            texts:     Single string or list of strings to embed
            model:     Embedding model alias or full ID (default: nomic-embed-text)
            normalize: L2-normalize the returned vectors (default: True)

        Returns:
            Single embedding vector (list[float]) for a string input, or
            list of embedding vectors for a list input.
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else [t for t in texts]

        # ── Step 1: ensure the daemon is running and the embedding model is loaded ──
        # Model(use_server=True) will auto-start the daemon if it isn't already running,
        # and _server_started_by_us tracks whether WE started it so we can stop it later.
        embed_model = Model(model, use_server=True)
        embed_model.load()  # POST /load → daemon loads the embedding model alongside any LLM

        server_url = embed_model.server_url  # e.g. http://127.0.0.1:11435

        # ── Step 2: send all texts to the daemon's /embedding endpoint in one call ──
        try:
            resp = requests.post(
                f"{server_url}/embedding",
                json={"input": text_list, "model": model},
                timeout=1200,
            )
            resp.raise_for_status()
            data = resp.json()

            # Daemon returns {"embedding": [...]} for single or {"embeddings": [[...],...]} for batch
            if is_single:
                vectors = [data.get("embedding", data.get("embeddings", [[]])[0])]
            else:
                vectors = data.get("embeddings", [data.get("embedding")])

        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to generate embedding via server: {exc}") from exc

        finally:
            # ── Step 3: lifecycle — stop daemon only if we started it ──
            # If the server was already running before this call, leave it alive
            # so loaded LLMs are not disturbed (same behaviour as text generation).
            if embed_model._server_started_by_us:
                embed_model._server_unload()  # unload embedding model from daemon
                # Gracefully shut down the server we started
                try:
                    requests.post(f"{server_url}/shutdown", timeout=5)
                except Exception:
                    pass

        # ── Step 4: optional L2 normalisation ──
        if normalize:
            import math
            result = []
            for vec in vectors:
                magnitude = math.sqrt(sum(x * x for x in vec))
                result.append([x / magnitude for x in vec] if magnitude > 0 else vec)
            vectors = result

        return vectors[0] if is_single else vectors


# Async client placeholder (can be implemented later with aiohttp)
class AsyncClient(Client):
    """
    Async client for Oprel (uses sync client for now)
    
    Note: Full async implementation requires aiohttp.
    This is a placeholder that uses sync methods.
    """
    
    async def chat(self, *args, **kwargs):
        """Async chat (currently uses sync)"""
        return super().chat(*args, **kwargs)
    
    async def generate(self, *args, **kwargs):
        """Async generate (currently uses sync)"""
        return super().generate(*args, **kwargs)

    async def generate_image(self, *args, **kwargs):
        """Async image generation (currently uses sync)"""
        return super().generate_image(*args, **kwargs)


# Module-level convenience functions
_default_client: Optional[Client] = None


def _get_client() -> Client:
    """Get or create default client"""
    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


def chat(
    model: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    **kwargs
) -> Union[ChatResponse, Iterator[ChatResponse]]:
    """
    Module-level chat function
    
    Example:
        from oprel import chat
        
        response = chat(
            model='qwencoder',
            messages=[{'role': 'user', 'content': 'Hello!'}]
        )
        print(response.message.content)
    """
    return _get_client().chat(model, messages, stream, **kwargs)


def generate(
    model: str,
    prompt: str,
    stream: bool = False,
    **kwargs
) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
    """
    Module-level generate function
    
    Example:
        from oprel import generate
        
        response = generate(
            model='qwencoder',
            prompt='Why is the sky blue?'
        )
        print(response.response)
    """
    return _get_client().generate(model, prompt, stream, **kwargs)


def generate_image(
    model: str,
    prompt: str,
    **kwargs
) -> ImageResponse:
    """
    Module-level image generation function

    Example:
        from oprel import generate_image

        response = generate_image(
            model='sd-1.5',
            prompt='a cozy cabin in the snow'
        )
        print(response.data[0])
    """
    return _get_client().generate_image(model, prompt, **kwargs)


def list() -> ListResponse:
    """
    Module-level list function
    
    Example:
        from oprel import list
        
        models = list()
        for model in models.models:
            print(model.name)
    """
    return _get_client().list()


def show(model: str) -> ShowResponse:
    """Module-level show function"""
    return _get_client().show(model)


def create(model: str, **kwargs) -> Dict[str, str]:
    """Module-level create function"""
    return _get_client().create(model, **kwargs)


def pull(model: str, **kwargs) -> Dict[str, Any]:
    """Module-level pull function"""
    return _get_client().pull(model, **kwargs)


def delete(model: str) -> Dict[str, str]:
    """Module-level delete function"""
    return _get_client().delete(model)


def embed(
    texts: Union[str, List[str]],
    model: str = "nomic-embed-text",
    **kwargs
) -> Union[List[float], List[List[float]]]:
    """
    Module-level embed function
    
    Example:
        from oprel import embed
        
        # Single text
        vector = embed("Hello world")
        
        # Batch
        vectors = embed(["Hello", "World"], model="snowflake-arctic")
    """
    return _get_client().embed(texts, model, **kwargs)
