from __future__ import annotations

import io
import os
from typing import Any

import modal

from modal_trellis2.core.preprocess import crop_to_foreground
from modal_trellis2.modal.app import app
from modal_trellis2.modal.image import cpu_runtime_image
from modal_trellis2.modal.volumes import MODEL_DIR, model_volume
from modal_trellis2.modal.weights import BIREFNET_LOCAL, BIREFNET_REPO

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
    """BiRefNet on CPU. GPU worker never loads this model."""

    @modal.enter()
    def setup(self) -> None:
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms

        model_volume.reload()
        local = f"{MODEL_DIR}/{BIREFNET_LOCAL}"
        source = local if os.path.isfile(os.path.join(local, "config.json")) else BIREFNET_REPO
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                source,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"BiRefNet is not on the Volume. Run `modal-trellis2 prefetch` (CPU). {exc}"
            ) from exc
        self.model.eval()
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
        return {"ok": self.model is not None, "device": "cpu", "model": BIREFNET_REPO}


def transforms_to_pil(mask):  # torch.Tensor
    from torchvision import transforms

    return transforms.ToPILImage()(mask)
