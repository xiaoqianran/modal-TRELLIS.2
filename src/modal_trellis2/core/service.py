from __future__ import annotations

from pathlib import Path

from modal_trellis2.core.config import (
    MAX_SEED,
    MIN_SEED,
    PIPELINES,
    TEXTURE_SIZES,
    Settings,
)
from modal_trellis2.core.generator import GenerateRequest, ImageTo3DGenerator
from modal_trellis2.core.glb import validate_glb
from modal_trellis2.core.image import encode_png, load_image
from modal_trellis2.core.jobs import Job, JobStore


class GenerateService:
    """Product contract: validated, bounded image bytes in; persisted GLB job out."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: JobStore | None = None,
        generator: ImageTo3DGenerator | None = None,
        dry_run: bool | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or JobStore(settings)
        if generator is None:
            raise ValueError("GenerateService requires an ImageTo3DGenerator")
        self.generator = generator
        self.dry_run = settings.dry_run if dry_run is None else dry_run

    def generate(
        self,
        image_bytes: bytes,
        *,
        filename: str = "input.png",
        seed: int | None = None,
        pipeline: str | None = None,
        texture_size: int = 1024,
        remesh: bool = True,
    ) -> Job:
        selected_pipeline = self.settings.default_pipeline if pipeline is None else pipeline
        if selected_pipeline not in PIPELINES:
            raise ValueError(
                f"unknown pipeline {selected_pipeline!r}; choose one of {', '.join(PIPELINES)}"
            )
        if texture_size not in TEXTURE_SIZES:
            allowed = ", ".join(str(value) for value in TEXTURE_SIZES)
            raise ValueError(f"unsupported texture_size {texture_size}; choose one of {allowed}")

        selected_seed = self.settings.default_seed if seed is None else seed
        if not MIN_SEED <= selected_seed <= MAX_SEED:
            raise ValueError(f"seed must be between {MIN_SEED} and {MAX_SEED}")

        # Caller errors are rejected before a Job exists. Keep a bounded lossless
        # local JobStore copy; the Modal adapter performs its own inline-safe JPEG transport.
        image = load_image(image_bytes)
        png = encode_png(image)

        job = self.store.create(
            filename=filename,
            seed=selected_seed,
            pipeline=selected_pipeline,
            dry_run=self.dry_run,
        )
        self.store.save_image(job, png)
        self.store.mark(job, "running")

        request = GenerateRequest(
            job_id=job.id,
            image_bytes=png,
            pipeline=selected_pipeline,
            seed=selected_seed,
            texture_size=texture_size,
            remesh=remesh,
        )
        result = None
        try:
            result = self.generator.generate(request)
            if result.error:
                raise RuntimeError(result.error)
            if not result.glb_bytes:
                raise RuntimeError("generator returned no GLB")
            validate_glb(result.glb_bytes)
            job.glb_filename = result.filename
            self.store.save_glb(job, result.glb_bytes)
        except Exception as exc:  # noqa: BLE001 - every runtime failure must close the Job
            return self.store.mark(
                job,
                "failed",
                error=str(exc),
                latency_ms=result.latency_ms if result is not None else 0.0,
                telemetry=result.telemetry if result is not None else {},
            )

        return self.store.mark(
            job,
            "completed",
            latency_ms=result.latency_ms,
            dry_run=result.dry_run,
            telemetry=result.telemetry,
            glb_filename=result.filename,
            glb_size_bytes=len(result.glb_bytes),
        )

    def generate_path(
        self,
        image_path: Path,
        output_path: Path,
        *,
        pipeline: str | None = None,
        seed: int | None = None,
        texture_size: int = 1024,
        remesh: bool = True,
    ) -> Job:
        source = Path(image_path)
        job = self.generate(
            source.read_bytes(),
            filename=source.name,
            pipeline=pipeline,
            seed=seed,
            texture_size=texture_size,
            remesh=remesh,
        )
        if job.status == "completed":
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.store.glb_path(job.id).read_bytes())
        return job


__all__ = ["GenerateService"]
