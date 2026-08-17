from __future__ import annotations

import modal

MODEL_DIR = "/models"

# Model artifacts are persistent on Modal. Generated GLBs remain owned by the local JobStore.
model_volume = modal.Volume.from_name("modal-trellis2-weights", create_if_missing=True)
