"""
Abstract base class for inference clients
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Union


class BaseClient(ABC):
    """
    Abstract interface for communicating with model backends.

    Implementations:
    - HTTPClient: Uses HTTP/REST (fallback, cross-platform)
    - UnixSocketClient: Uses Unix domain sockets (faster, Linux/Mac)
    - NamedPipeClient: Uses Windows named pipes (faster, Windows)
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = very random)
            stream: If True, return iterator of tokens; if False, return complete text
            **kwargs: Backend-specific parameters (top_p, top_k, stop sequences, etc.)

        Returns:
            If stream=False: Complete generated text as string
            If stream=True: Iterator yielding tokens as they're generated

        Raises:
            BackendError: If communication with backend fails
            MemoryError: If backend reports OOM
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the connection and clean up resources.
        Should be called when done with the client.
        """
        pass

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close()
