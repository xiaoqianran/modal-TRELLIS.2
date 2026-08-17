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
    MODELS_1024,
    MODELS_512,
    PRODUCTION_GPU,
    TRELLIS2_REPO,
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
    timeout=30 * 60,
    min_containers=GPU_MIN_CONTAINERS,
    max_containers=GPU_MAX_CONTAINERS,
    buffer_containers=GPU_BUFFER_CONTAINERS,
    scaledown_window=GPU_SCALEDOWN_SECONDS,
    retries=0,
    enable_memory_snapshot=True,
    env={
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": MODEL_DIR,
        "HF_HUB_DISABLE_TELEMETRY": "1",
    },
)
class Trellis2Worker:
    """Official TRELLIS.2. CPU snapshot loads weights; GPU only does .cuda() + run.

    Cost policy: this production worker has exactly one GPU container at a time.
    Bursty requests queue onto that container instead of scaling out to more GPUs.
    When the queue drains, the container scales to zero after the short idle window.
    """

    @modal.enter(snap=True)
    def load_cpu(self) -> None:
        """Read the Volume into CPU RAM. Modal snapshots this. No GPU here."""
        import sys

        model_volume.reload()
        if "/root/TRELLIS.2" not in sys.path:
            sys.path.insert(0, "/root/TRELLIS.2")
        _offline_env()
        weights = _require_local_weights()
        _use_local_dinov3()
        _skip_gpu_rembg()

        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        Trellis2ImageTo3DPipeline.model_names_to_load = list(MODELS_512)
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(weights)
        self.pipeline.rembg_model = None
        # 512-only weights fit these 80–96 GB GPUs. Keeping every stage resident
        # removes six CPU↔GPU transfers that dominated the earlier 178 s run.
        self.pipeline.low_vram = False
        self.weights_source = weights

    @modal.enter(snap=False)
    def move_to_gpu(self) -> None:
        """The only GPU-side setup: attach CUDA and move the already-loaded pipeline."""
        import o_voxel

        self.o_voxel = o_voxel
        self.pipeline.cuda()

    @modal.method()
    def health(self) -> dict[str, Any]:
        """Confirm the GPU container restored the snapshot and moved weights. Starts a GPU."""
        import torch

        weights = f"{MODEL_DIR}/trellis2"
        return {
            "ok": self.pipeline is not None,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "weights_local": os.path.exists(f"{weights}/pipeline.json"),
            "source": getattr(self, "weights_source", weights),
            "offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "low_vram": self.pipeline.low_vram,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            "scaledown_window": GPU_SCALEDOWN_SECONDS,
            "min_containers": GPU_MIN_CONTAINERS,
            "max_containers": GPU_MAX_CONTAINERS,
            "buffer_containers": GPU_BUFFER_CONTAINERS,
            "repo": TRELLIS2_REPO,
        }

    @modal.method()
    def warmup(self, image_bytes: bytes, seed: int = 42, pipeline_type: str = "512") -> dict[str, Any]:
        """Compile/warm CUDA kernels without running GLB postprocessing."""
        import time

        import torch
        from PIL import Image

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
            "offline": True,
            "scaledown_window": GPU_SCALEDOWN_SECONDS,
            "timings": {
                "infer_ms": infer_ms,
                "export_ms": export_ms,
            },
        }

    def _ensure_models(self, pipeline_type: str) -> None:
        needed = MODELS_512 if pipeline_type == "512" else MODELS_1024
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
