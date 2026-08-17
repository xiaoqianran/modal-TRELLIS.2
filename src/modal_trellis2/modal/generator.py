from __future__ import annotations

import time

from modal_trellis2.core.generator import GenerateRequest, GenerateResult
from modal_trellis2.core.preprocess import prepare_image
from modal_trellis2.modal.weights import PRODUCTION_GPU


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
                        "Run `uv run modal-trellis2 prefetch` then `uv run modal-trellis2 deploy`."
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
            if not isinstance(payload, dict):
                raise TypeError(f"worker returned {type(payload).__name__}, expected dict")
            glb_bytes = payload.get("glb_bytes")
            if not isinstance(glb_bytes, bytes) or not glb_bytes:
                raise TypeError("worker response is missing non-empty glb_bytes")
        except Exception as exc:  # noqa: BLE001
            return GenerateResult(
                job_id=request.job_id,
                error=(
                    f"Official TRELLIS.2-4B call failed: {exc}. "
                    "Run `uv run modal-trellis2 prefetch` then `uv run modal-trellis2 deploy`. "
                    "Use --dry-run only to test the upload/download loop."
                ),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        timings = dict(payload.get("timings") or {})
        timings["rembg_ms"] = rembg_ms
        timings["rembg"] = "cpu" if needs_rembg else "skipped-alpha"
        return GenerateResult(
            job_id=request.job_id,
            glb_bytes=glb_bytes,
            latency_ms=payload.get("latency_ms", (time.perf_counter() - started) * 1000),
            dry_run=False,
            telemetry={
                "backend": "official-trellis2",
                "model": payload.get("source") or "microsoft/TRELLIS.2-4B",
                "source_revision": payload.get("source_revision"),
                "model_revision": payload.get("model_revision"),
                "pipeline": payload.get("pipeline", request.pipeline),
                "seed": payload.get("seed", request.seed),
                "size_bytes": payload.get("size_bytes"),
                "gpu": PRODUCTION_GPU,
                "gpu_policy": "fixed-production-pool",
                "offline": payload.get("offline", True),
                "network_blocked": payload.get("network_blocked", True),
                "scaledown_window": payload.get("scaledown_window"),
                "container_instance_id": payload.get("container_instance_id"),
                "model_manifest": payload.get("model_manifest"),
                "timings": timings,
            },
        )
