from __future__ import annotations

import time

from modal_trellis2.core.generator import GenerateRequest, GenerateResult


class ModalTrellis2Generator:
    """Calls the deployed Trellis2Worker. Core stays free of Modal imports at rest."""

    def generate(self, request: GenerateRequest) -> GenerateResult:
        started = time.perf_counter()
        try:
            import modal

            from modal_trellis2.modal.app import APP_NAME
        except Exception as exc:  # noqa: BLE001
            return GenerateResult(
                job_id=request.job_id,
                error=f"Modal client import failed: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        try:
            cls = modal.Cls.from_name(APP_NAME, "Trellis2Worker")
            worker = cls.with_options(gpu=request.gpu)()
            payload = worker.generate.remote(
                request.image_bytes,
                seed=request.seed,
                pipeline_type=request.pipeline,
                texture_size=request.texture_size,
                remesh=request.remesh,
            )
        except Exception as exc:  # noqa: BLE001
            return GenerateResult(
                job_id=request.job_id,
                error=(
                    f"Official TRELLIS.2-4B call failed: {exc}. "
                    "Run `modal-trellis2 prefetch` then `modal-trellis2 deploy`. "
                    "Use --dry-run only to test the upload/download loop."
                ),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        return GenerateResult(
            job_id=request.job_id,
            glb_bytes=payload["glb_bytes"],
            latency_ms=payload.get("latency_ms", (time.perf_counter() - started) * 1000),
            dry_run=False,
            telemetry={
                "backend": "official-trellis2",
                "model": payload.get("source") or "microsoft/TRELLIS.2-4B",
                "pipeline": payload.get("pipeline", request.pipeline),
                "seed": payload.get("seed", request.seed),
                "size_bytes": payload.get("size_bytes"),
                "gpu": request.gpu,
            },
        )
