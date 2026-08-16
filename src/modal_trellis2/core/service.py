from __future__ import annotations

from modal_trellis2.core.config import PIPELINES, Settings
from modal_trellis2.core.generator import GenerateRequest, ImageTo3DGenerator
from modal_trellis2.core.glb import GlbError, validate_glb
from modal_trellis2.core.image import ImageError, encode_png, load_image
from modal_trellis2.core.jobs import Job, JobStore
from modal_trellis2.core.mock import MockGenerator


def build_service(settings: Settings, *, dry_run: bool | None = None) -> GenerateService:
    use_mock = settings.dry_run if dry_run is None else dry_run
    if use_mock:
        generator: ImageTo3DGenerator = MockGenerator()
    else:
        from modal_trellis2.modal.generator import ModalTrellis2Generator

        generator = ModalTrellis2Generator()
    return GenerateService(settings, generator=generator)


class GenerateService:
    """Owns the only product contract: image bytes in, GLB file out."""

    def __init__(
        self,
        settings: Settings,
        store: JobStore | None = None,
        generator: ImageTo3DGenerator | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or JobStore(settings)
        self.generator = generator or MockGenerator()

    def generate(
        self,
        image_bytes: bytes,
        *,
        filename: str = "input.png",
        seed: int | None = None,
        pipeline: str | None = None,
        dry_run: bool | None = None,
    ) -> Job:
        if pipeline is None:
            pipeline = self.settings.default_pipeline
        if pipeline not in PIPELINES:
            raise ValueError(f"unknown pipeline {pipeline!r}; choose one of {', '.join(PIPELINES)}")
        image = load_image(image_bytes)
        png = encode_png(image)
        job = self.store.create(
            filename=filename,
            seed=self.settings.default_seed if seed is None else seed,
            pipeline=pipeline,
            dry_run=self.settings.dry_run if dry_run is None else dry_run,
        )
        self.store.save_image(job, png)
        self.store.mark(job, "running")
        request = GenerateRequest(
            job_id=job.id,
            image_bytes=png,
            filename=job.filename,
            seed=job.seed,
            pipeline=job.pipeline,  # type: ignore[arg-type]
            gpu=self.settings.default_gpu,
        )
        try:
            result = self.generator.generate(request)
            if result.error:
                raise RuntimeError(result.error)
            if not result.glb_bytes:
                raise RuntimeError("generator returned no GLB")
            validate_glb(result.glb_bytes)
        except (ImageError, GlbError, RuntimeError, ValueError) as exc:
            return self.store.mark(job, "failed", error=str(exc))
        self.store.save_glb(job, result.glb_bytes)
        return self.store.mark(
            job,
            "completed",
            latency_ms=result.latency_ms,
            dry_run=result.dry_run,
            telemetry=result.telemetry,
            glb_filename=result.filename,
            glb_size_bytes=len(result.glb_bytes),
        )
