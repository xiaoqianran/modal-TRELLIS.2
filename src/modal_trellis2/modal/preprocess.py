from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import modal

from modal_trellis2.core.preprocess import crop_to_foreground
from modal_trellis2.modal.app import app
from modal_trellis2.modal.image import cpu_runtime_image
from modal_trellis2.modal.model_bundle import hf_model_ready
from modal_trellis2.modal.volumes import MODEL_DIR, model_volume
from modal_trellis2.modal.weights import BIREFNET_LOCAL, BIREFNET_REPO, RMBG_LOCAL, RMBG_REPO

# CPU rembg. This module must not import Trellis2Worker.


@app.cls(
    image=cpu_runtime_image,
    volumes={MODEL_DIR: model_volume},
    timeout=15 * 60,
    scaledown_window=300,
    retries=0,
    env={
        "HF_HOME": MODEL_DIR,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    },
)
class CpuPreprocessor:
    """Background removal on CPU. GPU worker never loads this model."""

    @modal.enter()
    def setup(self) -> None:
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        model_volume.reload()
        candidates = (
            (RMBG_REPO, Path(MODEL_DIR) / RMBG_LOCAL),
            (BIREFNET_REPO, Path(MODEL_DIR) / BIREFNET_LOCAL),
        )
        ready = [(repo, path) for repo, path in candidates if hf_model_ready(path)]
        if not ready:
            raise RuntimeError(
                "No complete background-removal model is on the Volume. "
                "Run `uv run modal-trellis2 prefetch` on CPU first."
            )

        failures: list[str] = []
        self.model = None
        self.model_repo = None
        self.model_source = None
        for repo, source in ready:
            try:
                model = AutoModelForImageSegmentation.from_pretrained(
                    str(source),
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{repo}: {type(exc).__name__}: {exc}")
                continue
            self.model = model.eval()
            self.model_repo = repo
            self.model_source = str(source)
            break

        if self.model is None:
            raise RuntimeError(
                "All complete local background-removal bundles failed to load: "
                + " | ".join(failures)
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @modal.method()
    def run(self, image_bytes: bytes) -> bytes:
        import torch
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        batch = self.transform(image).unsqueeze(0)
        with torch.inference_mode():
            mask = self.model(batch)[-1].sigmoid()[0].squeeze().cpu()
        alpha = transforms_to_pil(mask).resize(image.size)
        image.putalpha(alpha)
        cropped = crop_to_foreground(image)
        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")
        return buffer.getvalue()

    @modal.method()
    def health(self) -> dict[str, Any]:
        return {
            "ok": self.model is not None,
            "device": "cpu",
            "model": self.model_repo,
            "source": self.model_source,
        }


def transforms_to_pil(mask):  # torch.Tensor
    from torchvision import transforms

    return transforms.ToPILImage()(mask)
