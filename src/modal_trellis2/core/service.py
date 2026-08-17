from __future__ import annotations

from pathlib import Path

from modal_trellis2.core.generator import GenerateRequest
from modal_trellis2.core.ids import new_job_id
from modal_trellis2.core.jobs import JobRecord, JobStore


class GenerateService:
    def __init__(self, data_dir: Path, dry_run: bool = False) -> None:
        self.data_dir = Path(data_dir)
        self.store = JobStore(self.data_dir)
        self.dry_run = dry_run

    def _generator(self, dry_run: bool):
        if dry_run:
            from modal_trellis2.core.mock import MockGenerator

            return MockGenerator()
        from modal_trellis2.modal.generator import ModalTrellis2Generator

        return ModalTrellis2Generator()

    def generate_bytes(
        self,
        image_bytes: bytes,
        *,
        filename: str = "image.png",
        pipeline: str = "512",
        seed: int = 42,
        texture_size: int = 1024,
        remesh: bool = True,
        dry_run: bool | None = None,
    ) -> JobRecord:
        job_id = new_job_id()
        output_dir = self.data_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}.glb"
        effective_dry_run = self.dry_run if dry_run is None else dry_run

        job = JobRecord(
            id=job_id,
            status="running",
            input_name=filename,
            output_path=str(output_path),
            pipeline=pipeline,
            seed=seed,
            dry_run=effective_dry_run,
        )
        self.store.put(job)

        generator = self._generator(effective_dry_run)
        result = generator.generate(
            GenerateRequest(
                job_id=job_id,
                image_bytes=image_bytes,
                pipeline=pipeline,
                seed=seed,
                texture_size=texture_size,
                remesh=remesh,
            )
        )
        if not result.ok:
            job.status = "failed"
            job.error = result.error or "generation failed"
            job.latency_ms = result.latency_ms
            job.telemetry = result.telemetry
            self.store.put(job)
            return job

        output_path.write_bytes(result.glb_bytes or b"")
        job.status = "completed"
        job.latency_ms = result.latency_ms
        job.glb_size_bytes = output_path.stat().st_size
        job.telemetry = result.telemetry
        self.store.put(job)
        return job

    def generate_path(
        self,
        image_path: Path,
        output_path: Path,
        *,
        pipeline: str = "512",
        seed: int = 42,
        texture_size: int = 1024,
        remesh: bool = True,
    ) -> JobRecord:
        job = self.generate_bytes(
            Path(image_path).read_bytes(),
            filename=Path(image_path).name,
            pipeline=pipeline,
            seed=seed,
            texture_size=texture_size,
            remesh=remesh,
        )
        if job.status == "completed" and job.output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(job.output_path).read_bytes())
            job.output_path = str(target)
        return job
