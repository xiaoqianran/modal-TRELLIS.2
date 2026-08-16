from __future__ import annotations

import modal

MODEL_DIR = "/models"
RESULTS_DIR = "/results"

model_volume = modal.Volume.from_name("modal-trellis2-weights", create_if_missing=True)
results_volume = modal.Volume.from_name("modal-trellis2-results", create_if_missing=True)
