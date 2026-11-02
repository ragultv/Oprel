"""
Tee process for the server log: ``python -m oprel.utils.log_tee <log_file>``.

Reads the server's stdout/stderr (Python and native output alike) from stdin, echoes it to the
console and appends it to the log file. It runs as a separate process so it can still drain
the pipe and record the last lines when the server dies in native code (llama.cpp / CUDA abort).
"""

from __future__ import annotations

import os
import re
import signal
import sys
import time

_ANSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")
# llama.cpp CUDA-graph debug lines: one per decode, thousands per reply, no diagnostic value.
_NOISE = re.compile(
    rb"^(CUDA Graph id \d+ reused|ggml_backend_cuda_graph_compute: CUDA graph warmup (reset|complete))\s*$"
)
_SEGMENT = re.compile(rb"[^\r\n]*(?:\r\n|\n|\r)")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m oprel.utils.log_tee <log_file>", file=sys.stderr)
        return 2

    # Ctrl+C reaches every process on the console; stay alive so the server's shutdown is logged.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)

    console = sys.stdout.buffer
    src = sys.stdin.fileno()
    noise_dropped = 0

    with open(sys.argv[1], "ab", buffering=0) as log:

        def emit(segment: bytes) -> None:
            nonlocal noise_dropped
            line = segment.rstrip(b"\r\n")
            if _NOISE.match(line):
                noise_dropped += 1
                return
            try:
                console.write(segment)
                console.flush()
            except (OSError, ValueError):
                pass  # console closed; keep logging to the file
            # Carriage-return-only segments are progress-bar redraws: console only.
            if segment.endswith(b"\n"):
                log.write(time.strftime("%H:%M:%S ").encode() + _ANSI.sub(b"", line) + b"\n")

        pending = b""
        while True:
            try:
                chunk = os.read(src, 65536)
            except OSError:
                break
            if not chunk:
                break
            pending += chunk
            end = 0
            for m in _SEGMENT.finditer(pending):
                emit(m.group(0))
                end = m.end()
            pending = pending[end:]
        if pending:
            emit(pending + b"\n")
        log.write(
            time.strftime("%H:%M:%S ").encode()
            + f"[log] server output ended ({noise_dropped} CUDA-graph debug lines omitted)\n".encode()
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
