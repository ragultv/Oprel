from __future__ import annotations

import asyncio
import base64
import json
import time as time_module
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

from oprel.server.services.context import logger
from oprel.server.download_manager import download_manager
from oprel.server import db


@dataclass(frozen=True)
class DownloadStream:
    iterator: AsyncIterator[str]


def start_download(model_id: str, quantization: str | None) -> dict[str, Any]:
    from oprel.downloader.aliases import resolve_model_id
    from oprel.telemetry.recommender import recommend_quantization
    from oprel.core.config import Config
    import concurrent.futures

    resolved = resolve_model_id(model_id)

    if quantization:
        quant = quantization
        logger.info(f"Downloading model {resolved} with specified quantization: {quant}")
    else:
        quant = recommend_quantization()
        logger.info(f"Downloading model {resolved} with recommended quantization: {quant}")

    raw_id = f"{resolved}:{quant}:{uuid.uuid4().hex[:8]}"
    download_id = base64.urlsafe_b64encode(raw_id.encode()).decode().rstrip("=")

    download_manager.start_download(raw_id, resolved, quant)

    def download_task() -> None:
        from oprel.downloader.hub import download_model_with_progress

        config = Config()
        started = time_module.time()

        try:
            def progress_callback(downloaded: int, total: int) -> None:
                download_manager.update_progress(raw_id, downloaded, total)

            download_model_with_progress(
                resolved,
                quantization=quant,
                cache_dir=config.cache_dir,
                progress_callback=progress_callback,
            )
            download_manager.complete_download(raw_id)
            logger.info(f"Download completed: {raw_id}")

            progress = download_manager.get_progress(raw_id)
            db.save_download_log(
                model_id=resolved,
                model_name=model_id,
                quantization=quant,
                status="completed",
                size_bytes=progress.total_bytes if progress else 0,
                duration_seconds=round(time_module.time() - started, 2),
            )
        except Exception as exc:
            logger.error(f"Download failed: {raw_id} - {exc}")
            download_manager.fail_download(raw_id, str(exc))
            db.save_download_log(
                model_id=resolved,
                model_name=model_id,
                quantization=quant,
                status="error",
                duration_seconds=round(time_module.time() - started, 2),
                error=str(exc),
            )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    executor.submit(download_task)

    return {
        "success": True,
        "model_id": model_id,
        "quantization": quant,
        "download_id": download_id,
        "message": "Download started. Use /downloads/progress?id={download_id} to track progress.",
    }


def start_image_download(model_id: str, quantization: str | None = None) -> dict[str, Any]:
    from oprel.downloader.aliases import resolve_model_id
    from oprel.core.config import Config
    import concurrent.futures

    resolved = resolve_model_id(model_id)
    quant = quantization or "Q8_0"
    raw_id = f"{resolved}:GGUF:{uuid.uuid4().hex[:8]}"
    download_id = base64.urlsafe_b64encode(raw_id.encode()).decode().rstrip("=")

    download_manager.start_download(raw_id, resolved, "GGUF")

    def download_task() -> None:
        from oprel.downloader.hub import download_model_with_progress

        config = Config()
        started = time_module.time()

        try:
            def progress_callback(downloaded: int, total: int) -> None:
                download_manager.update_progress(raw_id, downloaded, total)

            download_model_with_progress(
                resolved,
                quantization=quant,
                cache_dir=config.cache_dir,
                progress_callback=progress_callback,
            )
            download_manager.complete_download(raw_id)

            progress = download_manager.get_progress(raw_id)
            db.save_download_log(
                model_id=resolved,
                model_name=model_id,
                quantization="GGUF",
                status="completed",
                size_bytes=progress.total_bytes if progress else 0,
                duration_seconds=round(time_module.time() - started, 2),
            )
        except Exception as exc:
            logger.error(f"Image download failed: {raw_id} - {exc}")
            download_manager.fail_download(raw_id, str(exc))
            db.save_download_log(
                model_id=resolved,
                model_name=model_id,
                quantization="GGUF",
                status="error",
                duration_seconds=round(time_module.time() - started, 2),
                error=str(exc),
            )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    executor.submit(download_task)

    return {
        "success": True,
        "model_id": model_id,
        "quantization": quant,
        "download_id": download_id,
        "message": "Image model download started. Use /downloads/progress?id={download_id} to track progress.",
    }


def _decode_download_id(download_id: str) -> str:
    padding = 4 - (len(download_id) % 4)
    if padding != 4:
        download_id += "=" * padding
    return base64.urlsafe_b64decode(download_id.encode()).decode()


def stream_download_progress(download_id: str) -> DownloadStream:
    raw_id = _decode_download_id(download_id)

    async def event_generator() -> AsyncIterator[str]:
        try:
            while True:
                progress = download_manager.get_progress(raw_id)

                if not progress:
                    yield f"data: {json.dumps({'error': 'Download not found'})}\n\n"
                    break

                data = {
                    "model_id": progress.model_id,
                    "quantization": progress.quantization,
                    "status": progress.status,
                    "progress": round(progress.progress, 2),
                    "downloaded": progress.downloaded_bytes,
                    "total": progress.total_bytes,
                    "speed": round(progress.speed_bps, 2),
                    "eta": round(progress.eta_seconds, 1),
                    "error": progress.error,
                }

                yield f"data: {json.dumps(data)}\n\n"

                if progress.status in ["completed", "error"]:
                    break

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info(f"Client disconnected from progress stream: {raw_id}")
        except Exception as exc:
            logger.error(f"Error streaming progress: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return DownloadStream(iterator=event_generator())


def list_downloads() -> dict[str, Any]:
    downloads = download_manager.get_all_downloads()
    return {
        "downloads": [
            {
                "download_id": download_id,
                "model_id": progress.model_id,
                "quantization": progress.quantization,
                "status": progress.status,
                "progress": round(progress.progress, 2),
                "downloaded": progress.downloaded_bytes,
                "total": progress.total_bytes,
                "speed": round(progress.speed_bps, 2),
                "eta": round(progress.eta_seconds, 1),
                "error": progress.error,
            }
            for download_id, progress in downloads.items()
        ]
    }


def list_download_logs(limit: int = 100) -> dict[str, Any]:
    return {"logs": db.list_download_logs(limit=limit)}


def get_download(download_id: str) -> dict[str, Any]:
    raw_id = _decode_download_id(download_id)
    progress = download_manager.get_progress(raw_id)
    if not progress:
        raise KeyError("Download not found")

    return {
        "download_id": download_id,
        "model_id": progress.model_id,
        "quantization": progress.quantization,
        "status": progress.status,
        "progress": round(progress.progress, 2),
        "downloaded": progress.downloaded_bytes,
        "total": progress.total_bytes,
        "speed": round(progress.speed_bps, 2),
        "eta": round(progress.eta_seconds, 1),
        "error": progress.error,
    }
