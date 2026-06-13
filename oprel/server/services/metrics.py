from __future__ import annotations

from oprel.server.domain.state import get_state


def get_metrics() -> dict[str, float | str | None]:
    import psutil
    from oprel.telemetry.hardware import get_vram_usage, detect_gpu

    state = get_state()

    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()

    vram = get_vram_usage()
    gpu = detect_gpu()

    return {
        "cpu_usage": cpu,
        "ram_total_gb": round(ram.total / (1024 ** 3), 2),
        "ram_used_gb": round(ram.used / (1024 ** 3), 2),
        "gpu_name": gpu["gpu_name"] if gpu else None,
        "gpu_usage": vram["utilization_percent"] if vram else None,
        "vram_total_mb": vram["total_mb"] if vram else None,
        "vram_used_mb": vram["used_mb"] if vram else None,
        "generation_speed": state.last_gen_speed,
    }
