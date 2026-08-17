from __future__ import annotations

import time
from pathlib import PurePosixPath

from modal_trellis2.core.generator import GenerateRequest, GenerateResult
from modal_trellis2.core.preprocess import prepare_image
from modal_trellis2.modal.volumes import OUTPUT_VOLUME_NAME
from modal_trellis2.modal.weights import PRODUCTION_GPU


class ModalTrellis2Generator:
    """CPU preflight/rembg first, then one deployed GPU worker.

    The network-blocked GPU writes large GLBs to a Modal Volume and returns only
    a small path/metadata payload. The local Modal client then downloads the file
    from that Volume and removes the remote copy after a verified read.
    """

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

        # Never discover a partial/mismatched Volume after the A100 has started.
        try:
            preflight_fn = modal.Function.from_name(APP_NAME, "prefetch_status")
            preflight = preflight_fn.remote()
            if not isinstance(preflight, dict):
                raise TypeError(
                    f"prefetch_status returned {type(preflight).__name__}, expected dict"
                )
            if not preflight.get("ok"):
                raise RuntimeError(f"offline model bundle is not ready: {preflight}")
        except Exception as exc:  # noqa: BLE001
            return GenerateResult(
                job_id=request.job_id,
                error=(
                    f"CPU preflight failed before GPU launch: {exc}. "
                    "Run `uv run modal-trellis2 prefetch` and confirm "
                    "`uv run modal-trellis2 prefetch --status` returns ok=true, then deploy."
                ),
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

        payload: dict[str, object]
        glb_bytes: bytes
        download_ms = 0.0
        cleanup_error: str | None = None
        try:
            # Production deliberately uses the deployed class configuration as-is.
            # Do not call `with_options(gpu=...)` here: every dynamic GPU option
            # creates its own autoscaling pool and can bypass the one-container cap.
            worker = modal.Cls.from_name(APP_NAME, "Trellis2Worker")()
            raw_payload = worker.generate.remote(
                image_bytes,
                job_id=request.job_id,
                seed=request.seed,
                pipeline_type=request.pipeline,
                texture_size=request.texture_size,
                remesh=request.remesh,
            )
            if not isinstance(raw_payload, dict):
                raise TypeError(f"worker returned {type(raw_payload).__name__}, expected dict")
            payload = raw_payload
            if payload.get("ok") is not True:
                remote_type = payload.get("error_type") or "RemoteError"
                remote_error = payload.get("error") or "unknown GPU worker error"
                raise RuntimeError(f"{remote_type}: {remote_error}")

            output_path = payload.get("output_path")
            if not isinstance(output_path, str):
                raise TypeError("worker response is missing output_path")
            path = PurePosixPath(output_path)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".glb":
                raise ValueError(f"worker returned unsafe output_path: {output_path!r}")

            output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME)
            download_started = time.perf_counter()
            glb_bytes = b"".join(output_volume.read_file(output_path))
            download_ms = (time.perf_counter() - download_started) * 1000
            if not glb_bytes:
                raise RuntimeError("output Volume returned an empty GLB")

            expected_size = payload.get("size_bytes")
            if isinstance(expected_size, int) and expected_size != len(glb_bytes):
                raise RuntimeError(
                    "output Volume size mismatch: "
                    f"expected {expected_size} bytes, downloaded {len(glb_bytes)}"
                )

            # The local JobStore owns the durable user copy. Delete the temporary
            # Modal Volume object after a verified download to avoid storage buildup.
            try:
                output_volume.remove_file(output_path)
            except Exception as exc:  # noqa: BLE001 - generation already succeeded
                cleanup_error = f"{type(exc).__name__}: {exc}"
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
        timings["download_ms"] = download_ms
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
                "output_transport": payload.get("output_transport", "modal-volume"),
                "output_cleanup_error": cleanup_error,
                "scaledown_window": payload.get("scaledown_window"),
                "container_instance_id": payload.get("container_instance_id"),
                "model_manifest": payload.get("model_manifest"),
                "timings": timings,
            },
        )
