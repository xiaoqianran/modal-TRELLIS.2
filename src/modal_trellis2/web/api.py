from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from modal_trellis2 import __version__
from modal_trellis2.core.config import PIPELINE_HINTS, PIPELINES, Settings, load_settings
from modal_trellis2.core.doctor import run_doctor
from modal_trellis2.core.image import ImageError
from modal_trellis2.core.jobs import JobStore
from modal_trellis2.core.service import GenerateService, build_service

router = APIRouter(prefix="/api")
_settings = load_settings()
_service = build_service(_settings)


def configure(settings: Settings | None = None, service: GenerateService | None = None) -> GenerateService:
    global _settings, _service
    _settings = settings or load_settings()
    _service = service or build_service(_settings)
    return _service


def service() -> GenerateService:
    return _service


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/meta")
def meta() -> dict[str, Any]:
    return {
        "version": __version__,
        "dry_run": _settings.dry_run,
        "defaults": {
            "pipeline": _settings.default_pipeline,
            "gpu": _settings.default_gpu,
            "seed": _settings.default_seed,
            "port": _settings.port,
        },
        "pipelines": [
            {"id": key, **value} for key, value in PIPELINE_HINTS.items() if key in PIPELINES
        ],
        "contract": {
            "in": "image/png|jpeg|webp",
            "out": "model/gltf-binary (.glb)",
        },
    }


@router.get("/doctor")
def doctor() -> dict[str, Any]:
    report = run_doctor(_settings)
    return {
        "ready": report.ready,
        "version": report.version,
        "checks": [check.__dict__ for check in report.checks],
    }


@router.post("/generate")
async def generate(
    image: UploadFile = File(...),
    seed: int = Form(42),
    pipeline: str = Form("512"),
    dry_run: bool | None = Form(None),
) -> dict[str, Any]:
    payload = await image.read()
    active = _service if dry_run is None else build_service(_settings, dry_run=dry_run)
    try:
        job = active.generate(
            payload,
            filename=image.filename or "input.png",
            seed=seed,
            pipeline=pipeline,
            dry_run=dry_run,
        )
    except (ImageError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if job.status == "failed":
        raise HTTPException(500, job.error or "generation failed")
    return job.public_dict()


@router.get("/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return [job.public_dict() for job in _service.store.list_jobs()]


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return _service.store.get(job_id).public_dict()
    except KeyError as exc:
        raise HTTPException(404, f"unknown job {job_id}") from exc


@router.get("/jobs/{job_id}/image")
def job_image(job_id: str) -> FileResponse:
    try:
        path = _service.store.image_path(job_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, f"no image for {job_id}") from exc
    return FileResponse(path, media_type="image/png", filename="input.png")


@router.get("/assets/{job_id}.glb")
def download_glb(job_id: str) -> Response:
    store: JobStore = _service.store
    try:
        job = store.get(job_id)
        path = store.glb_path(job_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, f"no GLB for {job_id}") from exc
    return Response(
        content=path.read_bytes(),
        media_type="model/gltf-binary",
        headers={"Content-Disposition": f'attachment; filename="{job.glb_filename}"'},
    )
