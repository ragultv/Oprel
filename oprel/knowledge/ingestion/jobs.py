"""
Durable ingestion job manager for restart-safe processing and retries.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from oprel.knowledge.models import IngestionJob, JobStatus

logger = logging.getLogger("oprel.knowledge.ingestion.jobs")


class JobManager:
    """
    Manages durable ingestion jobs with disk persistence and exponential backoff retry tracking.
    """

    def __init__(self, storage_dir: Path):
        self.storage_path = storage_dir / "ingestion_jobs.json"
        self._jobs: Dict[str, IngestionJob] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    job = IngestionJob(
                        id=item["id"],
                        document_id=item["document_id"],
                        status=JobStatus(item["status"]),
                        attempts=item.get("attempts", 0),
                        max_attempts=item.get("max_attempts", 3),
                        last_error=item.get("last_error"),
                        created_at=item.get("created_at", datetime.now(timezone.utc).isoformat()),
                        updated_at=item.get("updated_at", datetime.now(timezone.utc).isoformat()),
                    )
                    self._jobs[job.id] = job
        except Exception as e:
            logger.error(f"Failed to load ingestion jobs from {self.storage_path}: {e}")

    def _save(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = [
                {
                    "id": j.id,
                    "document_id": j.document_id,
                    "status": j.status.value,
                    "attempts": j.attempts,
                    "max_attempts": j.max_attempts,
                    "last_error": j.last_error,
                    "created_at": j.created_at,
                    "updated_at": j.updated_at,
                }
                for j in self._jobs.values()
            ]
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save ingestion jobs to {self.storage_path}: {e}")

    def create_job(self, document_id: str, job_id: Optional[str] = None) -> IngestionJob:
        jid = job_id or f"job_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{document_id[:8]}"
        job = IngestionJob(id=jid, document_id=document_id, status=JobStatus.PENDING)
        self._jobs[jid] = job
        self._save()
        return job

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
    ) -> Optional[IngestionJob]:
        job = self._jobs.get(job_id)
        if not job:
            return None

        job.status = status
        job.updated_at = datetime.now(timezone.utc).isoformat()
        if error:
            job.attempts += 1
            job.last_error = error
        self._save()
        return job

    def get_uncompleted_jobs(self) -> List[IngestionJob]:
        """Return pending or failed jobs eligible for resumption/retry."""
        return [
            j for j in self._jobs.values()
            if j.status in (JobStatus.PENDING, JobStatus.PROCESSING)
            or (j.status == JobStatus.FAILED and j.attempts < j.max_attempts)
        ]
