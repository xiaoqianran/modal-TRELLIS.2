from __future__ import annotations

import modal

from modal_trellis2.modal.image import cpu_image, trellis2_image
from modal_trellis2.modal.volumes import MODEL_DIR, RESULTS_DIR, model_volume, results_volume

APP_NAME = "modal-trellis2"

app = modal.App(APP_NAME)


def huggingface_secret() -> modal.Secret:
    return modal.Secret.from_name("huggingface-secret")


__all__ = [
    "APP_NAME",
    "MODEL_DIR",
    "RESULTS_DIR",
    "app",
    "huggingface_secret",
    "model_volume",
    "results_volume",
    "cpu_image",
    "trellis2_image",
]
