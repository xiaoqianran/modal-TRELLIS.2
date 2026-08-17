from __future__ import annotations

import time

from modal_trellis2.core.generator import GenerateRequest, GenerateResult
from modal_trellis2.core.preprocess import prepare_image


class ModalTrellis2Generator:
    """CPU rembg first, then the deployed GPU worker. Core stays free of Modal at rest."""

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
            image_bytes, needs_rembg = prepare_image(request.image_bytes)
        except Exception as exc:  # noqa: BLE001
            return GenerateResult(
                job_id=request.job_id,
                error=f"CPU image prepare failed: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        rembg_ms = 0.0
        if needs_rembg:
            rembg_started = time.perf_counter()
            try:
                cpu = modal.Cls.from_name(APP_NAME, "CpuPreprocessor")()
                image_bytes = cpu.run.remote(image_bytes)
            except Exception as exc:  # noqa: BLE001
                return GenerateResult(
                    job_id=request.job_id,
                    error=(
                        f"CPU rembg failed: {exc}. "
                        "Run `modal-trellis2 prefetch` then `modal-trellis2 deploy`."
                    ),
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            rembg_ms = (time.perf_counter() - rembg_started) * 1000

        try:
            # Production deliberately uses the deployed class configuration as-is.
            # Do not call `with_options(gpu=...)` here: every dynamic GPU option
            # creates its own autoscaling pool and can bypass the one-container cap.
            worker = modal.Cls.from_name(APP_NAME, "Trellis2Worker")()
            payload = worker.generate.remote(
                image_bytes,
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
        timings = dict(payload.get("timings") or {})
        timings["rembg_ms"] = rembg_ms
        timings["rembg"] = "cpu" if needs_rembg else "skipped-alpha"
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
                "gpu": "A100-80GB",
                "gpu_policy": "fixed-production-pool",
                "offline": payload.get("offline", True),
                "scaledown_window": payload.get("scaledown_window"),
                "timings": timings,
            },
        )
