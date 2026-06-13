"""Main user-facing model API."""

import threading
import subprocess
import sys
import time
from typing import Any, Iterator, Optional

import requests

from oprel.client.http import HTTPClient
from oprel.client.socket import UnixSocketClient
from oprel.core.config import Config
from oprel.core.exceptions import OprelError
from oprel.downloader.aliases import resolve_model_id
from oprel.downloader.hub import download_model
from oprel.runtime.monitor import ProcessMonitor
from oprel.runtime.process import ModelProcess
from oprel.telemetry.recommender import recommend_quantization
from oprel.utils.logging import get_logger

logger = get_logger(__name__)

# Default server configuration
DEFAULT_SERVER_URL = "http://127.0.0.1:11435"
SERVER_STARTUP_TIMEOUT = 30  # seconds


def _extract_model_size_from_name(model_id: str) -> int:
    """
    Extract model size in billions from model name.
    
    Examples:
        "bartowski/gemma-2-27b-it-GGUF" -> 27
        "TheBloke/Llama-2-7B-GGUF" -> 7
        "microsoft/phi-3-mini-3.8b" -> 4
    """
    import re
    
    # Look for patterns like "27b", "7B", "3.8b" etc
    match = re.search(r'(\d+(?:\.\d+)?)b', model_id.lower())
    if match:
        size = float(match.group(1))
        return int(round(size))
    
    # Fallback to 7B if can't detect
    return 7


class Model:
    """
    Main interface for loading and running local AI models.

    Usage (Server Mode - Default, Fast after first load):
        >>> from oprel import Model
        >>> model = Model("TheBloke/Llama-2-7B-GGUF")
        >>> response = model.generate("What is Python?")
        >>> print(response)

    Usage (Direct Mode - Current behavior):
        >>> from oprel import Model
        >>> model = Model("TheBloke/Llama-2-7B-GGUF", use_server=False)
        >>> response = model.generate("What is Python?")
        >>> print(response)
    """

    def __init__(
        self,
        model_id: str,
        quantization: Optional[str] = None,
        max_memory_mb: Optional[int] = None,
        backend: str = "llama.cpp",
        config: Optional[Config] = None,
        use_server: bool = True,
        server_url: str = DEFAULT_SERVER_URL,
        allow_low_quality: bool = False,
    ):
        """
        Initialize a model instance.

        Args:
            model_id: HuggingFace model ID (e.g., "TheBloke/Llama-2-7B-GGUF")
            quantization: Quantization level (Q4_K_M, Q5_K_M, Q8_0) or None for auto
            max_memory_mb: Maximum memory limit in MB (None for auto)
            backend: Backend engine. Only ``llama.cpp`` is supported.
            config: Custom configuration object
            use_server: Whether to use persistent server mode (default=True)
            server_url: URL of the oprel daemon server (default="http://127.0.0.1:11434")
            allow_low_quality: Allow low-quality quantizations (Q2_K) if needed
        """
        self.model_id = resolve_model_id(model_id)  # Resolve aliases like 'llama3' to full path
        self.config = config or Config()
        self.use_server = use_server
        self.server_url = server_url.rstrip("/")
        
        if backend != "llama.cpp":
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                "This build only supports the llama.cpp backend."
            )

        self.backend_name = "llama.cpp"

        # Initialize runtime state early to prevent __del__ errors
        self._process: Optional[ModelProcess] = None
        self._monitor: Optional[ProcessMonitor] = None
        self._client: Optional[HTTPClient | UnixSocketClient] = None
        self._model_path = None
        self._lock = threading.Lock()
        self._loaded = False
        self._server_started_by_us = False

        # Detect model size from name
        model_size_b = _extract_model_size_from_name(self.model_id)
        
        # Auto-detect quantization if not specified
        if quantization is None:
            # First, check for locally available quantizations
            from oprel.utils.model_info import get_local_quantizations
            local_quants = get_local_quantizations(self.model_id, self.config.cache_dir)
            
            if local_quants:
                # Use the first local quantization (prefer higher quality)
                # Sort by quality: Q8_0 > Q6_K > Q5_K_M > Q4_K_M > Q3_K_M > Q2_K
                quality_order = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
                for quant in quality_order:
                    if quant in local_quants:
                        quantization = quant
                        logger.info(f"Using locally available quantization: {quantization}")
                        break
                
                # Fallback to first local quant if none match quality order
                if quantization is None:
                    quantization = local_quants[0]
                    logger.info(f"Using locally available quantization: {quantization}")
            else:
                # No local quantizations, recommend one
                quantization = recommend_quantization(model_size_b=model_size_b, allow_low_quality=allow_low_quality)
                logger.info(f"No local quantizations found, auto-selected: {quantization}")

        self.quantization = quantization
        self.max_memory_mb = max_memory_mb or self.config.default_max_memory_mb
        
    def _is_server_running(self) -> bool:
        """Check if the oprel daemon server is running."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _auto_start_server(self) -> bool:
        """
        Start the oprel daemon server if not running.
        
        Returns:
            True if server is running (was already running or started successfully)
        """
        if self._is_server_running():
            return True
        
        logger.info("Oprel daemon not running, starting automatically...")
        
        try:
            # Start server in background subprocess
            # Use pythonw on Windows to avoid console window, python otherwise
            python_exe = sys.executable
            
            # Start the server as a detached subprocess
            if sys.platform == "win32":
                # Windows: use CREATE_NEW_PROCESS_GROUP, DETACHED_PROCESS, and CREATE_NO_WINDOW
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                subprocess.Popen(
                    [python_exe, "-m", "oprel.server.daemon"],
                    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
            else:
                # Unix: use nohup-like behavior
                subprocess.Popen(
                    [python_exe, "-m", "oprel.server.daemon"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            
            # Wait for server to start
            start_time = time.time()
            while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
                if self._is_server_running():
                    logger.info("Oprel daemon started successfully")
                    self._server_started_by_us = True
                    return True
                time.sleep(0.5)
            
            logger.warning("Failed to start oprel daemon within timeout")
            return False
            
        except Exception as e:
            logger.warning(f"Failed to auto-start oprel daemon: {e}")
            return False

    def _server_load(self) -> None:
        """Load model via server API."""
        payload = {
            "model_id": self.model_id,
            "quantization": self.quantization,
            "max_memory_mb": self.max_memory_mb,
            "backend": self.backend_name,
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/load",
                json=payload,
                timeout=600,  # Model loading can take several minutes
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("success"):
                logger.info(f"Model registered with server: {result.get('message')}")
            else:
                raise OprelError(f"Server failed to load model: {result}")
                
        except requests.RequestException as e:
            raise OprelError(f"Failed to communicate with server: {e}") from e

    def _server_generate(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        stream: bool = False,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        reset_conversation: bool = False,
        thinking: bool = False,
        rag: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """Generate text via server API."""
        payload = {
            "model_id": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "conversation_id": conversation_id,
            "system_prompt": system_prompt,
            "reset_conversation": reset_conversation,
            "thinking": thinking,
            "rag": rag,
        }
        
        try:
            if stream:
                return self._server_stream_generate(payload)
            else:
                response = requests.post(
                    f"{self.server_url}/generate",
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
                return result.get("text", "")
                
        except requests.RequestException as e:
            raise OprelError(f"Failed to generate via server: {e}") from e

    def _server_stream_generate(self, payload: dict) -> Iterator[str]:
        """Stream generation via server API."""
        try:
            response = requests.post(
                f"{self.server_url}/generate",
                json=payload,
                stream=True,
                timeout=300,
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                
                if line_str.startswith("data: "):
                    data = line_str[6:]  # Don't strip - preserve spaces in tokens
                    if data.strip() == "[DONE]":
                        return
                    yield data
                    
        except requests.RequestException as e:
            raise OprelError(f"Failed to stream from server: {e}") from e

    def _server_unload(self) -> None:
        """Unload model via server API."""
        try:
            # Use POST with JSON body to handle model IDs with special characters
            response = requests.post(
                f"{self.server_url}/unload",
                json={"model_id": self.model_id},
                timeout=30,
            )
            
            if response.status_code == 404:
                logger.warning(f"Model {self.model_id} was not loaded on server")
                return
                
            response.raise_for_status()
            logger.info(f"Model unloaded from server: {self.model_id}")
            
        except requests.RequestException as e:
            logger.warning(f"Failed to unload from server: {e}")

    def load(self) -> None:
        """
        Download and load the model into memory.
        Pre-warming step to avoid latency on first generation.
        """
        with self._lock:
            if self._loaded:
                logger.warning("Model already loaded")
                return

            # CRITICAL: Validate model type before attempting to load
            # This prevents trying to load diffusion models with llama.cpp
            from oprel.models.model_types import detect_model_type, is_supported_model_type, get_unsupported_message
            
            model_type = detect_model_type(self.model_id)
            
            if not is_supported_model_type(model_type):
                error_msg = get_unsupported_message(model_type)
                logger.error(f"Unsupported model type: {model_type}")
                raise OprelError(
                    f"Cannot load model '{self.model_id}': {model_type} models are not supported.\n\n"
                    f"{error_msg}"
                )

            if self.use_server:
                # Server mode: ensure server is running and register model
                if not self._auto_start_server():
                    logger.warning("Server unavailable, falling back to direct mode")
                    self._load_direct()
                else:
                    self._server_load()
                    self._loaded = True
            else:
                # Direct mode: spawn subprocess locally
                self._load_direct()

    def _load_direct(self) -> None:
        """Load model directly (current behavior)."""
        # Step 1: Download model if needed
        logger.info(f"Downloading model: {self.model_id}")
        model_path = download_model(
            self.model_id,
            quantization=self.quantization,
            cache_dir=self.config.cache_dir,
        )
        self._model_path = model_path

        logger.info(f"Starting {self.backend_name} backend")
        self._process = ModelProcess(
            model_path=model_path,
            backend=self.backend_name,
            config=self.config,
        )
        self._process.start()

        # Step 4: Start health monitor
        self._monitor = ProcessMonitor(
            process=self._process.process,
            max_memory_mb=self.max_memory_mb,
        )
        self._monitor.start()

        # Step 5: Initialize client
        # Unix sockets only work on Linux/macOS, not Windows
        import platform

        use_socket = (
            self.config.use_unix_socket
            and self._process.socket_path
            and platform.system() != "Windows"
        )

        if use_socket:
            self._client = UnixSocketClient(self._process.socket_path)
        else:
            self._client = HTTPClient(self._process.port)

        self._loaded = True
        logger.info(f"Model loaded successfully on port {self._process.port}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        stream: bool = False,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        reset_conversation: bool = False,
        images: Optional[list] = None,  # New: Support for vision models
        thinking: bool = False,
        rag: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            stream: Whether to stream tokens incrementally
            conversation_id: ID for conversation history (Server mode only)
            system_prompt: System prompt to use (Server mode only)
            reset_conversation: Reset conversation history (Server mode only)
            images: Optional list of base64-encoded images for vision models
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text (string if stream=False, iterator if stream=True)

        Raises:
            OprelError: If model is not loaded
            MemoryError: If model exceeds memory limit
        """
        if not self._loaded:
            logger.info("Model not loaded, loading now...")
            self.load()

        if self.use_server and self._is_server_running():
            # Server mode
            return self._server_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                conversation_id=conversation_id,
                system_prompt=system_prompt,
                reset_conversation=reset_conversation,
                thinking=thinking,
                rag=rag,
                **kwargs,
            )
        else:
            # Direct mode
            # Check health before generation
            health_error = self._monitor.check_health()
            if health_error:
                raise health_error

            # --- Manual mode injection for direct mode ---
            from oprel.utils.chat_templates import THINKING_MODE_INSTRUCTION, FAST_MODE_SUPPRESSION
            
            # Note: We apply basic instruction injection here. 
            # If the user uses the CLI, it already called format_chat_prompt(..., thinking=T/F),
            # so we avoid double-injection by checking if the markers already exist.
            if thinking:
                if THINKING_MODE_INSTRUCTION.strip() not in prompt:
                    prompt = THINKING_MODE_INSTRUCTION + "\n" + prompt
                
                # Thinking mode needs budget
                if max_tokens < 8192:
                    max_tokens = 8192
                    logger.debug("Direct mode: bumping max_tokens to 8192 for thinking")
            else:
                # Fast mode: actively suppress and cap
                if FAST_MODE_SUPPRESSION.strip() not in prompt:
                    prompt = FAST_MODE_SUPPRESSION + "\n" + prompt
                
                if max_tokens > 2048:
                    max_tokens = 2048
                    logger.debug("Direct mode: capping max_tokens to 2048 for fast mode")

            # Generate via client
            return self._client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                images=images,
                model=self.model_id,
                **kwargs,
            )


    def generate_structured(
        self,
        prompt: str,
        response_model: Any,
        max_retries: int = 2,
        **kwargs: Any
    ) -> Any:
        """
        Generate structured data matching a Pydantic model.
        
        Args:
            prompt: Input text
            response_model: Pydantic BaseModel class
            max_retries: Number of retries if validation fails
            **kwargs: Arguments passed to generate()
            
        Returns:
            Instance of response_model
            
        Raises:
            ValidationError: If model fails to generate valid data after retries
        """
        import json
        import re
        from pydantic import ValidationError
        
        schema = response_model.model_json_schema()
        
        # enhance prompt with schema compliance instructions
        system_instruction = (
            f"You are a precise data extraction engine.\n"
            f"You MUST output valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            f"Do not include markdown formatting (```json), comments, or explanations.\n"
            f"Output ONLY the JSON object."
        )
        
        # If user passed system prompt, append to it, otherwise use ours
        user_system = kwargs.get('system_prompt', '')
        full_system = f"{user_system}\n\n{system_instruction}" if user_system else system_instruction
        kwargs['system_prompt'] = full_system
        
        # Force low temperature for deterministic structure
        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0.2
            
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Generate raw text
                text = self.generate(
                    prompt=prompt,
                    stream=False,
                    **kwargs
                )
                
                # Clean up markdown code blocks if present
                clean_text = text.strip()
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean_text, re.DOTALL)
                if match:
                    clean_text = match.group(1)
                else:
                    # Try to find the first { and last }
                    start = clean_text.find('{')
                    end = clean_text.rfind('}')
                    if start != -1 and end != -1:
                        clean_text = clean_text[start : end + 1]
                
                # Parse JSON
                data = json.loads(clean_text)
                
                # Validate with Pydantic
                return response_model.model_validate(data)
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Structured geneation attempt {attempt+1} failed: {e}")
                last_error = e
                # Setup retry prompt
                prompt += f"\n\nERROR: The previous output was invalid. {str(e)}\nPlease correct the JSON format."
                
        raise last_error

    def unload(self) -> None:
        """
        Stop the model process and free resources.
        
        Note: In server mode, this does NOT unload from the server.
        Server-mode models stay cached for fast subsequent requests.
        Only direct-mode models are actually unloaded.
        """
        with self._lock:
            if not self._loaded:
                return

            if self.use_server:
                # Server mode: Do NOT unload from server
                # The whole point of server mode is persistent caching
                logger.debug(f"Model {self.model_id} kept in server cache")
            else:
                # Direct mode: stop local process or PyTorch backend
                if self._monitor:
                    self._monitor.stop()

                if self._process:
                    self._process.stop()

                logger.info("Model unloaded")

            self._loaded = False

    def __enter__(self):
        """Context manager support"""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.unload()

    def __del__(self):
        """Cleanup on garbage collection.
        
        Note: In server mode, we do NOT unload from the server on __del__.
        The whole point of server mode is to keep models cached persistently.
        Only direct mode models get cleaned up on garbage collection.
        """
        if self._loaded and not self.use_server:
            # Only cleanup direct mode models
            self.unload()

