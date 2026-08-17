from __future__ import annotations

import io
import json
import os
from typing import Any

import modal

from modal_trellis2.modal.app import app
from modal_trellis2.modal.image import trellis2_image
from modal_trellis2.modal.volumes import MODEL_DIR, model_volume
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
    """Point the official image encoder at the CPU-prefetched folder."""
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

    local = f"{MODEL_DIR}/{DINOV3_LOCAL}"
    original = DinoV3FeatureExtractor.__init__

    def patched(self, model_name: str, image_size: int = 512):  # type: ignore[no-untyped-def]
        if os.path.isfile(os.path.join(local, "config.json")):
            model_name = local
        original(self, model_name, image_size)

    DinoV3FeatureExtractor.__init__ = patched  # type: ignore[method-assign]


def _skip_gpu_rembg() -> None:
    """Do not construct BiRefNet in the GPU container; CPU already removed background."""
    from trellis2.pipelines import rembg

    def no_op(self, *args: Any, **kwargs: Any) -> None:
        return None

    rembg.BiRefNet.__init__ = no_op


def _require_local_weights() -> str:
    weights = f"{MODEL_DIR}/trellis2"
    if not os.path.exists(f"{weights}/pipeline.json"):
        raise RuntimeError(
            "Official TRELLIS.2-4B is not on the Volume. "
            "Run `modal-trellis2 prefetch` on CPU first. GPU will not download."
        )
    return weights


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
        """Load the offline pipeline only after Modal has attached the A100."""
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
        weights = _require_local_weights()

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
        self.pipeline.cuda()

        self.o_voxel = o_voxel
        self.weights_source = weights
        self.container_instance_id = uuid.uuid4().hex

    @modal.method()
    def health(self) -> dict[str, Any]:
        """Confirm the GPU container loaded local weights and attached CUDA. Starts a GPU."""
        import torch

        weights = f"{MODEL_DIR}/trellis2"
        return {
            "ok": self.pipeline is not None,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "weights_local": os.path.exists(f"{weights}/pipeline.json"),
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
            "production_pipelines": list(PRODUCTION_PIPELINES),
            "container_instance_id": self.container_instance_id,
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
        self._ensure_models(pipeline_type)
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
        target = decimation_target or (500_000 if pipeline_type == "512" else 1_000_000)
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

    def _read_model_manifest(self) -> dict[str, Any] | None:
        path = f"{MODEL_DIR}/manifest.json"
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _ensure_models(self, pipeline_type: str) -> None:
        self._require_production_pipeline(pipeline_type)
        needed = MODELS_512
        missing = [name for name in needed if name not in self.pipeline.models]
        if not missing:
            return
        from trellis2 import models

        weights = getattr(self, "weights_source", _require_local_weights())
        with open(f"{weights}/pipeline.json", encoding="utf-8") as handle:
            spec = json.load(handle)["args"]["models"]
        for name in missing:
            self.pipeline.models[name] = models.from_pretrained(f"{weights}/{spec[name]}")
            self.pipeline.models[name].eval()
