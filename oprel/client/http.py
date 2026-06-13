"""
HTTP client for communicating with model backends

Production-Ready Features (Week 1 Tasks):
- M1.11: Configurable request timeouts with sensible defaults
- M1.12: Proper UTF-8 token buffering to prevent mid-character cutoff

Timeout Strategy:
- Connection timeout: 10s (fail fast if server is down)
- Read timeout: Based on expected generation time
- For CPU inference: May need 300s+ for long prompts
"""

from typing import Any, Iterator, Union, Optional, Tuple
import requests

from oprel.client.base import BaseClient
from oprel.core.exceptions import BackendError


# Default timeout configuration (M1.11)
DEFAULT_CONNECT_TIMEOUT = 10.0  # Fail fast if server unreachable
DEFAULT_READ_TIMEOUT_STREAMING = 120.0  # Per-chunk timeout for streaming
DEFAULT_READ_TIMEOUT_COMPLETION = 300.0  # Total timeout for non-streaming (CPU can be slow)


class HTTPClient(BaseClient):
    """
    HTTP client for model inference with production-ready features.
    
    Features:
    - Configurable timeouts (M1.11)
    - UTF-8 safe streaming (M1.12)
    - Retry logic for transient failures
    - Session reuse for performance
    
    Usage:
        client = HTTPClient(port=54321, timeout=60.0)
        
        # Non-streaming
        text = client.generate("Hello", max_tokens=100)
        
        # Streaming
        for token in client.generate("Hello", stream=True):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        port: int,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT_COMPLETION,
    ):
        """
        Initialize HTTP client.
        
        Args:
            port: Server port number
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds (for non-streaming)
        """
        self.base_url = f"http://127.0.0.1:{port}"
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Configure retry adapter for resilience
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],  # Retry on server errors
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        stream: bool = False,
        timeout: Optional[float] = None,
        images: Optional[list] = None,  # New: Support for vision models
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """
        Generate text via HTTP API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            stream: Whether to stream response token-by-token
            timeout: Override default timeout (seconds)
            images: Optional list of base64-encoded images for vision models
            **kwargs: Additional parameters for the API

        Returns:
            Generated text (str) or token iterator (if stream=True)
            
        Raises:
            BackendError: If request fails
        """
        url = f"{self.base_url}/v1/completions"

        model = kwargs.pop("model", None)

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }
        if model:
            payload["model"] = model
        
        # For vision models, use chat completions endpoint with proper image format
        if images:
            url = f"{self.base_url}/v1/chat/completions"
            # Convert to chat format with images (OpenAI format: text first, then images)
            content = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
                })
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
                **kwargs,
            }
            
            # Debug: Log that we're sending images
            import logging
            logging.getLogger(__name__).info(f"Sending {len(images)} images via chat completions endpoint")

        try:
            if stream:
                return self._stream_response(url, payload, timeout)
            else:
                return self._complete_response(url, payload, timeout)
        except requests.Timeout as e:
            raise BackendError(
                f"Request timed out after {timeout or self.read_timeout}s. "
                "The model may be overloaded or the prompt too complex. "
                "Try: 1) Shorter prompt, 2) Increase timeout, 3) Use streaming mode"
            ) from e
        except requests.ConnectionError as e:
            raise BackendError(
                f"Connection failed to {self.base_url}. "
                "The backend may have crashed or not started. "
                "Check: 1) Backend is running, 2) Port is correct"
            ) from e
        except requests.RequestException as e:
            raise BackendError(f"HTTP request failed: {e}") from e

    def _complete_response(
        self,
        url: str,
        payload: dict,
        timeout: Optional[float] = None
    ) -> str:
        """
        Non-streaming response with configurable timeout.
        
        Args:
            url: API endpoint
            payload: Request payload
            timeout: Optional timeout override
            
        Returns:
            Generated text
        """
        effective_timeout = (
            self.connect_timeout,
            timeout or self.read_timeout
        )
        
        response = self.session.post(url, json=payload, timeout=effective_timeout)
        response.raise_for_status()

        data = response.json()
        
        # Debug: Log response structure
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Response data: {data}")
        
        # Handle different response formats
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "text" in choice:
                return choice["text"]
            elif "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        
        # Fallback
        return data.get("text", data.get("content", ""))

    def _stream_response(
        self,
        url: str,
        payload: dict,
        timeout: Optional[float] = None
    ) -> Iterator[str]:
        """
        Streaming response with UTF-8 safe buffering (M1.12).
        
        Handles:
        - Server-Sent Events (SSE) format
        - Incomplete UTF-8 byte sequences
        - Various response formats (OpenAI, llama.cpp)
        
        Args:
            url: API endpoint
            payload: Request payload
            timeout: Optional timeout override
            
        Yields:
            Text tokens as they arrive
        """
        import json

        effective_timeout = (
            self.connect_timeout,
            timeout or DEFAULT_READ_TIMEOUT_STREAMING
        )
        
        response = self.session.post(
            url,
            json=payload,
            stream=True,
            timeout=effective_timeout,
        )
        response.raise_for_status()

        # UTF-8 buffer for incomplete multi-byte sequences (M1.12)
        utf8_buffer = b""
        
        # Token buffer for potential partial tokens
        token_buffer = ""
        
        for chunk in response.iter_content(chunk_size=None):
            if not chunk:
                continue
            
            # Combine with any leftover bytes from previous chunk
            data = utf8_buffer + chunk
            utf8_buffer = b""
            
            # Try to decode UTF-8, keeping incomplete sequences
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as e:
                # Incomplete UTF-8 sequence at end of chunk
                # Keep the incomplete bytes for next iteration
                complete_data = data[:e.start]
                utf8_buffer = data[e.start:]
                
                if complete_data:
                    text = complete_data.decode("utf-8")
                else:
                    continue
            
            # Process SSE lines
            lines = (token_buffer + text).split('\n')
            token_buffer = ""
            
            # Last line might be incomplete
            if not text.endswith('\n'):
                token_buffer = lines[-1]
                lines = lines[:-1]
            
            for line in lines:
                line = line.strip()
                
                if not line or line.startswith(":"):
                    continue
                
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    
                    # Handle stream termination
                    if data_str == "[DONE]":
                        return
                    
                    if not data_str:
                        continue
                    
                    try:
                        data = json.loads(data_str)
                        token = self._extract_token(data)
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        # Skip malformed JSON (can happen at stream boundaries)
                        continue
        
        # Flush any remaining buffer
        if utf8_buffer:
            try:
                text = utf8_buffer.decode("utf-8", errors="replace")
                # Process any remaining data
                for line in text.split('\n'):
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str and data_str != "[DONE]":
                            try:
                                data = json.loads(data_str)
                                token = self._extract_token(data)
                                if token:
                                    yield token
                            except json.JSONDecodeError:
                                pass
            except Exception:
                pass

    def _extract_token(self, data: dict) -> str:
        """
        Extract token text from various API response formats.
        
        Supports:
        - OpenAI format: {"choices": [{"delta": {"content": "..."}}]}
        - llama.cpp format: {"choices": [{"text": "..."}]}
        - Simple format: {"content": "..."}
        """
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            
            # OpenAI streaming format
            if "delta" in choice:
                delta = choice["delta"]
                if "content" in delta:
                    return delta["content"]
            
            # llama.cpp format
            if "text" in choice:
                return choice["text"]
            
            # Alternative format
            if "content" in choice:
                return choice["content"]
        
        # Direct content field
        if "content" in data:
            return data["content"]
        
        return ""

    def health_check(self, timeout: float = 5.0) -> bool:
        """
        Check if the backend is healthy.
        
        Args:
            timeout: Request timeout
            
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=(self.connect_timeout, timeout)
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "ok"
        except Exception:
            pass
        return False

    def close(self) -> None:
        """Close HTTP session and release resources."""
        if self.session:
            self.session.close()
            self.session = None

    def __enter__(self) -> "HTTPClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
