"""
Unix domain socket client for low-latency IPC
"""

import socket
import json
from pathlib import Path
from typing import Any, Iterator, Union

from oprel.client.base import BaseClient
from oprel.core.exceptions import BackendError
from oprel.utils.logging import get_logger

logger = get_logger(__name__)


class UnixSocketClient(BaseClient):
    """
    Unix domain socket client for fast local IPC.

    ~5-10x lower latency than HTTP for small messages.
    Only works on Linux/macOS (not Windows).
    """

    def __init__(self, socket_path: Path, timeout: float = 60.0):
        """
        Initialize Unix socket client.

        Args:
            socket_path: Path to Unix socket file (e.g., /tmp/oprel.sock)
            timeout: Socket timeout in seconds
        """
        self.socket_path = Path(socket_path)
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._connect()

    def _connect(self) -> None:
        """Establish socket connection"""
        try:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect(str(self.socket_path))
            pass  # Connected successfully
        except (FileNotFoundError, ConnectionRefusedError) as e:
            raise BackendError(
                f"Failed to connect to socket {self.socket_path}. " f"Is the model server running?"
            ) from e
        except Exception as e:
            raise BackendError(f"Socket connection error: {e}") from e

    def generate(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """
        Generate text via Unix socket.

        Protocol:
        1. Send JSON request with newline terminator
        2. Receive JSON response(s)
        3. For streaming: multiple JSON objects, one per line
        """
        if not self._socket:
            raise BackendError("Socket not connected")

        # Build request payload (OpenAI-compatible format)
        request = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **kwargs,
        }

        try:
            # Send request as JSON with newline delimiter
            request_data = json.dumps(request) + "\n"
            self._socket.sendall(request_data.encode("utf-8"))

            if stream:
                return self._stream_response()
            else:
                return self._complete_response()

        except socket.timeout:
            raise BackendError(
                f"Request timed out after {self.timeout}s. "
                f"Model may be overloaded or prompt too long."
            )
        except BrokenPipeError:
            raise BackendError("Connection lost. Model process may have crashed.")
        except Exception as e:
            raise BackendError(f"Socket communication error: {e}") from e

    def _complete_response(self) -> str:
        """Receive complete (non-streaming) response"""
        buffer = b""

        while True:
            chunk = self._socket.recv(4096)
            if not chunk:
                break

            buffer += chunk

            # Check if we have a complete JSON object (ends with newline)
            if b"\n" in buffer:
                break

        try:
            response = json.loads(buffer.decode("utf-8").strip())

            # Handle error responses
            if "error" in response:
                raise BackendError(f"Backend error: {response['error']}")

            # Extract text from OpenAI-compatible response
            if "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["text"]

            raise BackendError(f"Unexpected response format: {response}")

        except json.JSONDecodeError as e:
            raise BackendError(f"Invalid JSON response: {e}") from e

    def _stream_response(self) -> Iterator[str]:
        """Receive streaming response (yields tokens)"""
        buffer = b""

        while True:
            try:
                chunk = self._socket.recv(1024)
                if not chunk:
                    break

                buffer += chunk

                # Process complete lines (each line = one JSON object)
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)

                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line.decode("utf-8"))

                        # Handle stream termination
                        if data.get("done"):
                            return

                        # Handle errors
                        if "error" in data:
                            raise BackendError(f"Stream error: {data['error']}")

                        # Yield token
                        if "choices" in data and len(data["choices"]) > 0:
                            token = data["choices"][0].get("text", "")
                            if token:
                                yield token

                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line: {line}")
                        continue

            except socket.timeout:
                logger.warning("Stream timeout, ending generation")
                break

    def close(self) -> None:
        """Close socket connection"""
        if self._socket:
            try:
                self._socket.close()
                pass  # Socket closed
            except Exception as e:
                logger.warning(f"Error closing socket: {e}")
            finally:
                self._socket = None

    def __del__(self):
        """Cleanup on garbage collection"""
        self.close()
