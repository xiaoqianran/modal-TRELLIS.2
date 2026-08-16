from __future__ import annotations

import io
import os
from typing import Any

import modal

from modal_trellis2.modal.app import app, huggingface_secret
from modal_trellis2.modal.image import trellis2_image
from modal_trellis2.modal.volumes import MODEL_DIR, RESULTS_DIR, model_volume, results_volume


@app.cls(
    gpu="A100-80GB",
    image=trellis2_image,
    volumes={MODEL_DIR: model_volume, RESULTS_DIR: results_volume},
    secrets=[huggingface_secret()],
    timeout=30 * 60,
    scaledown_window=10,
)
class Trellis2Worker:
    """Official TRELLIS.2 image-to-3D. Keep this file the only place that imports trellis2."""

    @modal.enter()
    def setup(self) -> None:
        import sys

        from huggingface_hub import login

        model_volume.reload()
        if "/root/TRELLIS.2" not in sys.path:
            sys.path.insert(0, "/root/TRELLIS.2")
        os.environ["HF_HOME"] = MODEL_DIR
        os.environ["HF_HUB_CACHE"] = f"{MODEL_DIR}/cache"
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token, add_to_git_credential=False)

        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        import o_voxel

        weights = f"{MODEL_DIR}/trellis2"
        source = weights if os.path.exists(f"{weights}/pipeline.json") else "microsoft/TRELLIS.2-4B"
        self.weights_source = source
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(source)
        self.pipeline.cuda()
        self.o_voxel = o_voxel

    @modal.method()
    def health(self) -> dict[str, Any]:
        """Confirm the GPU container loaded the official pipeline. Starts a GPU."""
        import torch

        weights = f"{MODEL_DIR}/trellis2"
        return {
            "ok": self.pipeline is not None,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "weights_local": os.path.exists(f"{weights}/pipeline.json"),
            "source": getattr(self, "weights_source", weights),
        }

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        seed: int = 42,
        pipeline_type: str = "512",
        texture_size: int = 1024,
        remesh: bool = True,
    ) -> dict[str, Any]:
        import time
        from PIL import Image

        started = time.perf_counter()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        mesh = self.pipeline.run(image, seed=seed, pipeline_type=pipeline_type)[0]
        mesh.simplify(16_777_216)
        glb = self.o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=500_000 if pipeline_type == "512" else 1_000_000,
            texture_size=texture_size,
            remesh=remesh,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        buffer = io.BytesIO()
        glb.export(buffer, file_type="glb")
        payload = buffer.getvalue()
        job_path = f"{RESULTS_DIR}/last.glb"
        with open(job_path, "wb") as handle:
            handle.write(payload)
        results_volume.commit()
        return {
            "glb_bytes": payload,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "pipeline": pipeline_type,
            "seed": seed,
            "size_bytes": len(payload),
            "source": getattr(self, "weights_source", f"{MODEL_DIR}/trellis2"),
        }
