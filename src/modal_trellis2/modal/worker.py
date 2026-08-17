from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any

import modal

from modal_trellis2.modal.app import app
from modal_trellis2.modal.image import trellis2_image
from modal_trellis2.modal.model_bundle import require_hf_model_dir, require_trellis_bundle
from modal_trellis2.modal.volumes import MODEL_DIR, OUTPUT_DIR, model_volume, output_volume
from modal_trellis2.modal.weights import (
    DINOV3_LOCAL,
    GPU_BUFFER_CONTAINERS,
    GPU_MAX_CONTAINERS,
    GPU_MIN_CONTAINERS,
    GPU_SCALEDOWN_SECONDS,
    GPU_TIMEOUT_SECONDS,
    MODELS_512,
    PRODUCTION_GPU,
    PRODUCTION_PIPELINES,
    PRODUCTION_TEXTURE_SIZES,
    TRELLIS2_MODEL_REVISION,
    TRELLIS2_REPO,
    TRELLIS2_SOURCE_REVISION,
)

# GPU is offline. Weights come from the CPU Volume. Do not import this
# module from prefetch.py — `modal run -m prefetch` must stay CPU-only.
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def _offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = MODEL_DIR
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _use_local_dinov3() -> None:
    """Point the official image encoder at the fully validated CPU-prefetched folder."""
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

    local = f"{MODEL_DIR}/{DINOV3_LOCAL}"
    original = DinoV3FeatureExtractor.__init__

    def patched(self, model_name: str, image_size: int = 512):  # type: ignore[no-untyped-def]
        model_name = local
        original(self, model_name, image_size)

    DinoV3FeatureExtractor.__init__ = patched  # type: ignore[method-assign]


def _skip_gpu_rembg() -> None:
    """Do not construct BiRefNet in the GPU container; CPU already removed background."""
    from trellis2.pipelines import rembg

    def no_op(self, *args: Any, **kwargs: Any) -> None:
        return None

    rembg.BiRefNet.__init__ = no_op


def _require_local_weights() -> tuple[str, dict[str, Any]]:
    weights = Path(MODEL_DIR) / "trellis2"
    bundle = require_trellis_bundle(weights)
    require_hf_model_dir(Path(MODEL_DIR) / DINOV3_LOCAL, label="DINOv3")
    return str(weights), bundle


def _output_relative_path(job_id: str) -> str:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ValueError("invalid job_id for output path")
    return f"jobs/{job_id}/mesh.glb"


@app.cls(
    gpu=PRODUCTION_GPU,
    image=trellis2_image,
    volumes={MODEL_DIR: model_volume, OUTPUT_DIR: output_volume},
    timeout=GPU_TIMEOUT_SECONDS,
    min_containers=GPU_MIN_CONTAINERS,
    max_containers=GPU_MAX_CONTAINERS,
    buffer_containers=GPU_BUFFER_CONTAINERS,
    scaledown_window=GPU_SCALEDOWN_SECONDS,
    retries=0,
    block_network=True,
    env={
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": MODEL_DIR,
        "HF_HUB_DISABLE_TELEMETRY": "1",
    },
)
class Trellis2Worker:
    """Official TRELLIS.2 worker initialized only after the GPU is attached.

    TRELLIS.2 imports flex_gemm/Triton while importing the pipeline. A normal
    Modal CPU memory-snapshot phase has no GPU driver, so importing TRELLIS in
    ``@modal.enter(snap=True)`` fails before weights can load. Keep snapshots
    disabled here and initialize the local, prefetched pipeline in the regular
    GPU lifecycle hook instead.

    Large GLBs are written to a temporary Modal Volume so the Function result is
    always a small metadata object. The local client downloads and verifies the
    file, persists it in JobStore, then removes the temporary remote copy.

    Catchable Python initialization failures are stored as ``init_error`` instead
    of escaping this lifecycle hook. Methods then return a readable error instead
    of turning a bad model import/load into a deployed-container crash loop.
    """

    @modal.enter()
    def setup_gpu(self) -> None:
        """Initialize after CUDA attach without crash-looping on catchable Python errors."""
        import gc
        import uuid

        import torch

        self.container_instance_id = uuid.uuid4().hex
        self.pipeline = None
        self.o_voxel = None
        self.weights_source = f"{MODEL_DIR}/trellis2"
        self.bundle_validation = None
        self.init_error = None
        self.vram_after_load = None

        try:
            self._initialize_gpu()
        except Exception as exc:  # noqa: BLE001 - lifecycle must not crash-loop on Python errors
            self.init_error = f"{type(exc).__name__}: {exc}"
            self.pipeline = None
            self.o_voxel = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _initialize_gpu(self) -> None:
        import sys

        import o_voxel
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Trellis2Worker requires CUDA before TRELLIS.2 is imported")

        model_volume.reload()
        if "/root/TRELLIS.2" not in sys.path:
            sys.path.insert(0, "/root/TRELLIS.2")
        _offline_env()
        weights, bundle_validation = _require_local_weights()

        # These imports transitively initialize flex_gemm/Triton. They must stay
        # after the CUDA availability check; CPU-only memory snapshots have no
        # active Triton driver and fail at import time.
        _use_local_dinov3()
        _skip_gpu_rembg()
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        Trellis2ImageTo3DPipeline.model_names_to_load = list(MODELS_512)
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(weights)
        pipeline.rembg_model = None
        pipeline.low_vram = False
        self.pipeline = pipeline
        self._require_loaded_models()
        pipeline.cuda()
        torch.cuda.synchronize()

        self.o_voxel = o_voxel
        self.weights_source = weights
        self.bundle_validation = bundle_validation
        self.vram_after_load = self._vram_stats()

    @modal.method()
    def health(self) -> dict[str, Any]:
        """Return GPU initialization state without converting init errors into crash loops."""
        import torch

        cuda = torch.cuda.is_available()
        ready = self.init_error is None and self.pipeline is not None and self.o_voxel is not None
        return {
            "ok": ready,
            "init_error": self.init_error,
            "cuda": cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "weights_local": bool((self.bundle_validation or {}).get("ok")),
            "source": self.weights_source,
            "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "network_blocked": True,
            "output_transport": "modal-volume",
            "low_vram": getattr(self.pipeline, "low_vram", None),
            "vram": {
                "after_load": self.vram_after_load,
                "current": self._vram_stats(),
            },
            "scaledown_window": GPU_SCALEDOWN_SECONDS,
            "timeout_seconds": GPU_TIMEOUT_SECONDS,
            "min_containers": GPU_MIN_CONTAINERS,
            "max_containers": GPU_MAX_CONTAINERS,
            "buffer_containers": GPU_BUFFER_CONTAINERS,
            "repo": TRELLIS2_REPO,
            "source_revision": TRELLIS2_SOURCE_REVISION,
            "model_revision": TRELLIS2_MODEL_REVISION,
            "production_pipelines": list(PRODUCTION_PIPELINES),
            "container_instance_id": self.container_instance_id,
            "bundle_validation": self.bundle_validation,
            "model_manifest": self._read_model_manifest(),
        }

    @modal.method()
    def warmup(
        self,
        image_bytes: bytes,
        seed: int = 42,
        pipeline_type: str = "512",
    ) -> dict[str, Any]:
        """Compile/warm CUDA kernels without running GLB postprocessing."""
        import time

        import torch
        from PIL import Image

        self._require_ready()
        self._require_production_pipeline(pipeline_type)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        torch.cuda.reset_peak_memory_stats()
        vram_before = self._vram_stats()
        started = time.perf_counter()
        mesh = self.pipeline.run(
            image,
            seed=seed,
            pipeline_type=pipeline_type,
            preprocess_image=False,
        )[0]
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        vram_after = self._vram_stats()
        vertices = int(mesh.vertices.shape[0])
        faces = int(mesh.faces.shape[0])
        del mesh
        torch.cuda.empty_cache()
        return {
            "ok": True,
            "latency_ms": elapsed_ms,
            "vertices": vertices,
            "faces": faces,
            "pipeline": pipeline_type,
            "vram": {"before": vram_before, "after": vram_after},
        }

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        job_id: str,
        seed: int = 42,
        pipeline_type: str = "512",
        texture_size: int = 1024,
        remesh: bool = True,
        decimation_target: int | None = None,
    ) -> dict[str, Any]:
        """Generate one GLB and return only small metadata."""
        import time

        started = time.perf_counter()
        try:
            return self._generate_impl(
                image_bytes=image_bytes,
                job_id=job_id,
                seed=seed,
                pipeline_type=pipeline_type,
                texture_size=texture_size,
                remesh=remesh,
                decimation_target=decimation_target,
                started=started,
            )
        except Exception as exc:  # noqa: BLE001 - keep remote error payload portable/small
            return {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "container_instance_id": self.container_instance_id,
                "vram": self._vram_stats(),
            }

    def _generate_impl(
        self,
        *,
        image_bytes: bytes,
        job_id: str,
        seed: int,
        pipeline_type: str,
        texture_size: int,
        remesh: bool,
        decimation_target: int | None,
        started: float,
    ) -> dict[str, Any]:
        import time

        import torch
        from PIL import Image

        self._require_ready()
        self._require_production_request(pipeline_type, texture_size)
        relative_output = _output_relative_path(job_id)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        torch.cuda.reset_peak_memory_stats()
        vram_before_infer = self._vram_stats()
        infer_started = time.perf_counter()
        mesh = self.pipeline.run(
            image,
            seed=seed,
            pipeline_type=pipeline_type,
            preprocess_image=False,
        )[0]
        torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - infer_started) * 1000
        vram_after_infer = self._vram_stats()
        mesh.simplify(16_777_216)

        export_started = time.perf_counter()
        target = decimation_target or 500_000
        glb = self.o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=target,
            texture_size=texture_size,
            remesh=remesh,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        buffer = io.BytesIO()
        glb.export(buffer, file_type="glb")
        torch.cuda.synchronize()
        payload = buffer.getvalue()
        payload_size = len(payload)
        export_ms = (time.perf_counter() - export_started) * 1000
        vram_after_export = self._vram_stats()

        persist_started = time.perf_counter()
        output_path = Path(OUTPUT_DIR) / relative_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(".glb.tmp")
        tmp_path.write_bytes(payload)
        os.replace(tmp_path, output_path)
        output_volume.commit()
        persist_ms = (time.perf_counter() - persist_started) * 1000

        del mesh, glb, buffer, payload
        torch.cuda.empty_cache()
        vram_after_cleanup = self._vram_stats()

        return {
            "ok": True,
            "output_path": relative_output,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "pipeline": pipeline_type,
            "seed": seed,
            "size_bytes": payload_size,
            "decimation_target": target,
            "texture_size": texture_size,
            "remesh": remesh,
            "source": self.weights_source,
            "source_revision": TRELLIS2_SOURCE_REVISION,
            "model_revision": TRELLIS2_MODEL_REVISION,
            "offline": True,
            "network_blocked": True,
            "output_transport": "modal-volume",
            "scaledown_window": GPU_SCALEDOWN_SECONDS,
            "container_instance_id": self.container_instance_id,
            "model_manifest": self._read_model_manifest(),
            "vram": {
                "after_load": self.vram_after_load,
                "before_infer": vram_before_infer,
                "after_infer": vram_after_infer,
                "after_export": vram_after_export,
                "after_cleanup": vram_after_cleanup,
            },
            "timings": {
                "infer_ms": infer_ms,
                "export_ms": export_ms,
                "persist_ms": persist_ms,
            },
        }

    def _vram_stats(self) -> dict[str, float | None]:
        import torch

        if not torch.cuda.is_available():
            return {
                "allocated_gb": None,
                "reserved_gb": None,
                "peak_allocated_gb": None,
                "peak_reserved_gb": None,
                "free_gb": None,
                "total_gb": None,
            }
        free, total = torch.cuda.mem_get_info()
        gib = 2**30
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / gib, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / gib, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / gib, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / gib, 3),
            "free_gb": round(free / gib, 3),
            "total_gb": round(total / gib, 3),
        }

    def _require_ready(self) -> None:
        if self.init_error:
            raise RuntimeError(f"GPU initialization failed: {self.init_error}")
        if self.pipeline is None or self.o_voxel is None:
            raise RuntimeError("GPU worker initialized without a usable TRELLIS.2 pipeline")
        self._require_loaded_models()

    def _require_production_request(self, pipeline_type: str, texture_size: int) -> None:
        self._require_production_pipeline(pipeline_type)
        if texture_size not in PRODUCTION_TEXTURE_SIZES:
            allowed = ", ".join(str(value) for value in PRODUCTION_TEXTURE_SIZES)
            raise ValueError(
                f"texture_size {texture_size} is not enabled in production; allowed: {allowed}"
            )

    def _require_production_pipeline(self, pipeline_type: str) -> None:
        if pipeline_type not in PRODUCTION_PIPELINES:
            allowed = ", ".join(PRODUCTION_PIPELINES)
            raise ValueError(
                f"pipeline {pipeline_type!r} is not enabled in production; allowed: {allowed}"
            )

    def _require_loaded_models(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("TRELLIS.2 pipeline is not initialized")
        missing = [name for name in MODELS_512 if name not in self.pipeline.models]
        if missing:
            raise RuntimeError(
                "Pinned TRELLIS.2 pipeline did not load all production models: "
                + ", ".join(missing)
                + ". Re-run CPU prefetch; do not late-load models inside a running GPU container."
            )

    def _read_model_manifest(self) -> dict[str, Any] | None:
        path = f"{MODEL_DIR}/manifest.json"
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else None
        except (OSError, json.JSONDecodeError):
            return None
