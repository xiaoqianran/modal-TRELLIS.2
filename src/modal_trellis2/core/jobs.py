from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from modal_trellis2.core.config import Settings
from modal_trellis2.core.ids import new_id

JobStatus = Literal["pending", "running", "completed", "failed"]


class Job(BaseModel):
    id: str
    status: JobStatus = "pending"
    created_at: str
    completed_at: str | None = None
    filename: str = "input.png"
    seed: int = 42
    pipeline: str = "512"
    dry_run: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    glb_filename: str = "mesh.glb"
    glb_size_bytes: int = 0
    telemetry: dict[str, Any] = Field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        payload = self.model_dump()
        payload["asset_url"] = f"/api/assets/{self.id}.glb" if self.status == "completed" else None
        payload["image_url"] = f"/api/jobs/{self.id}/image"
        return payload


class JobStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.jobs_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, *, filename: str, seed: int, pipeline: str, dry_run: bool) -> Job:
        job = Job(
            id=new_id("job"),
            created_at=_now(),
            filename=Path(filename).name or "input.png",
            seed=seed,
            pipeline=pipeline,
            dry_run=dry_run,
        )
        self._write(job)
        return job

    def get(self, job_id: str) -> Job:
        path = self._meta_path(job_id, create=False)
        if not path.is_file():
            raise KeyError(job_id)
        return Job.model_validate_json(path.read_text(encoding="utf-8"))

    def list_jobs(self, limit: int | None = None) -> list[Job]:
        jobs = [
            Job.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self.root.glob("*/meta.json")
        ]
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs if limit is None else jobs[: max(0, limit)]

    def save_image(self, job: Job, image_bytes: bytes) -> Path:
        path = self._dir(job.id) / "input.png"
        path.write_bytes(image_bytes)
        return path

    def save_glb(self, job: Job, glb_bytes: bytes) -> Path:
        path = self._dir(job.id) / job.glb_filename
        path.write_bytes(glb_bytes)
        job.glb_size_bytes = len(glb_bytes)
        self._write(job)
        return path

    def glb_path(self, job_id: str) -> Path:
        job = self.get(job_id)
        path = self.root / job_id / job.glb_filename
        if not path.is_file():
            raise FileNotFoundError(job_id)
        return path

    def image_path(self, job_id: str) -> Path:
        self.get(job_id)
        path = self.root / job_id / "input.png"
        if not path.is_file():
            raise FileNotFoundError(job_id)
        return path

    def mark(self, job: Job, status: JobStatus, **updates: Any) -> Job:
        job.status = status
        for key, value in updates.items():
            setattr(job, key, value)
        if status in {"completed", "failed"}:
            job.completed_at = _now()
        self._write(job)
        return job

    def _dir(self, job_id: str) -> Path:
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _meta_path(self, job_id: str, *, create: bool = True) -> Path:
        base = self._dir(job_id) if create else self.root / job_id
        return base / "meta.json"

    def _write(self, job: Job) -> None:
        path = self._meta_path(job.id)
        tmp = path.with_suffix(".json.tmp")
        payload = json.dumps(job.model_dump(), indent=2)
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
