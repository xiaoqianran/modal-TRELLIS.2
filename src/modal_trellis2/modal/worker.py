from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import modal

from modal_trellis2.modal.app import app
from modal_trellis2.modal.image import trellis2_image
from modal_trellis2.modal.model_bundle import require_hf_model_dir, require_trellis_bundle
from modal_trellis2.modal.volumes import MODEL_DIR, model_volume
from modal_trellis2.modal.weights import (
    DINOV3_LOCAL,
    GPU_BUFFER_CONTAINERS,
    GPU_MAX_CONTAINERS,
    GPU_MIN_CONTAINERS,
    GPU_SCALEDOWN_SECONDS,
    GPU_TIMEOUT_SECONDS,
    MAX_MODAL_RESULT_BYTES,
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


@app.cls(
    gpu=PRODUCTION_GPU,
    image=trellis2_image,
    volumes={MODEL_DIR: model_volume},
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

    Cost policy: this production worker has exactly one GPU container at a time.
    Bursty requests queue onto that container instead of scaling out to more GPUs.
    When the queue drains, the container scales to zero after the short idle window.
    """

    @modal.enter()
    def setup_gpu(self) -> None:
        """Validate the offline bundle, then import TRELLIS after CUDA is attached."""
        import sys
        import uuid

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
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(weights)
        self.pipeline.rembg_model = None
        self.pipeline.low_vram = False
        self._require_loaded_models()
        self.pipeline.cuda()

        self.o_voxel = o_voxel
        self.weights_source = weights
        self.bundle_validation = bundle_validation
        self.container_instance_id = uuid.uuid4().hex

    @modal.method()
    def health(self) -> dict[str, Any]:
        """Confirm the GPU container loaded the complete pinned bundle. Starts a GPU."""
        import torch

        weights = f"{MODEL_DIR}/trellis2"
        return {
            "ok": self.pipeline is not None,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "weights_local": bool(getattr(self, "bundle_validation", {}).get("ok")),
            "source": getattr(self, "weights_source", weights),
            "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "network_blocked": True,
            "low_vram": self.pipeline.low_vram,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
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

        self._require_production_pipeline(pipeline_type)
        self._require_loaded_models()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        started = time.perf_counter()
        mesh = self.pipeline.run(
            image,
            seed=seed,
            pipeline_type=pipeline_type,
            preprocess_image=False,
        )[0]
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
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
        }

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        seed: int = 42,
        pipeline_type: str = "512",
        texture_size: int = 1024,
        remesh: bool = True,
        decimation_target: int | None = None,
    ) -> dict[str, Any]:
        import time

        from PIL import Image

        started = time.perf_counter()
        self._require_production_request(pipeline_type, texture_size)
        self._require_loaded_models()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        infer_started = time.perf_counter()
        mesh = self.pipeline.run(
            image,
            seed=seed,
            pipeline_type=pipeline_type,
            preprocess_image=False,
        )[0]
        infer_ms = (time.perf_counter() - infer_started) * 1000
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
        payload = buffer.getvalue()
        if len(payload) > MAX_MODAL_RESULT_BYTES:
            raise RuntimeError(
                "Generated GLB exceeds the Modal RPC safety limit "
                f"({len(payload)} > {MAX_MODAL_RESULT_BYTES} bytes). "
                "Reduce texture_size or decimation target."
            )
        export_ms = (time.perf_counter() - export_started) * 1000
        return {
            "glb_bytes": payload,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "pipeline": pipeline_type,
            "seed": seed,
            "size_bytes": len(payload),
            "decimation_target": target,
            "texture_size": texture_size,
            "remesh": remesh,
            "source": getattr(self, "weights_source", f"{MODEL_DIR}/trellis2"),
            "source_revision": TRELLIS2_SOURCE_REVISION,
            "model_revision": TRELLIS2_MODEL_REVISION,
            "offline": True,
            "network_blocked": True,
            "scaledown_window": GPU_SCALEDOWN_SECONDS,
            "container_instance_id": self.container_instance_id,
            "model_manifest": self._read_model_manifest(),
            "timings": {
                "infer_ms": infer_ms,
                "export_ms": export_ms,
            },
        }

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
