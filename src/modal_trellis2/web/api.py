from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from modal_trellis2.core.config import settings
from modal_trellis2.core.jobs import JobStore
from modal_trellis2.core.service import GenerateService

router = APIRouter(prefix="/api")
store = JobStore(settings.data_dir)
service = GenerateService(data_dir=settings.data_dir, dry_run=settings.dry_run)


@router.post("/generate")
async def generate(
    image: UploadFile = File(...),
    pipeline: str = Form("512"),
    seed: int = Form(42),
    texture_size: int = Form(1024),
    remesh: bool = Form(True),
    dry_run: bool = Form(False),
):
    """Generate with the fixed production GPU pool; requests cannot choose GPU type."""
    if not image.filename:
        raise HTTPException(status_code=400, detail="missing image filename")
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty image")

    result = service.generate_bytes(
        payload,
        filename=image.filename,
        pipeline=pipeline,
        seed=seed,
        texture_size=texture_size,
        remesh=remesh,
        dry_run=dry_run,
    )
    if result.error:
        raise HTTPException(status_code=500, detail=result.error)
    return result.model_dump(mode="json")


@router.get("/assets/{asset_id}.glb")
def asset(asset_id: str) -> FileResponse:
    job = store.get(asset_id)
    if job is None or not job.output_path:
        raise HTTPException(status_code=404, detail="asset not found")
    path = Path(job.output_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="asset file missing")
    return FileResponse(path, media_type="model/gltf-binary", filename=f"{asset_id}.glb")


@router.get("/jobs")
def jobs(limit: int = 50):
    return [job.model_dump(mode="json") for job in store.list(limit=limit)]
