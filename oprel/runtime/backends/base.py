"""
Abstract base class for backend implementations
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from oprel.core.config import Config


class BaseBackend(ABC):
    """
    Abstract interface for model backends.
    All backends (e.g., llama.cpp) must implement this.
    """

    def __init__(
        self,
        binary_path: Path,
        model_path: Path,
        config: Config,
    ):
        self.binary_path = binary_path
        self.model_path = model_path
        self.config = config

    @abstractmethod
    def build_command(self, port: int) -> List[str]:
        """
        Build the command-line arguments to spawn the backend.

        Args:
            port: Port number for the server

        Returns:
            List of command arguments (e.g., ["./llama-server", "--model", "..."])
        """
        pass

    @abstractmethod
    def get_api_format(self) -> str:
        """
        Get the API format this backend uses.

        Returns:
            "openai" or "custom"
        """
        pass
