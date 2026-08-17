from __future__ import annotations

import modal

MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"
MODEL_VOLUME_NAME = "modal-trellis2-weights"
OUTPUT_VOLUME_NAME = "modal-trellis2-results"

# Model weights are persistent and shared by CPU prefetch + the GPU worker.
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

# The GPU writes generated GLBs here temporarily and returns only small metadata.
# The local client downloads the verified file, persists it in JobStore, then removes
# this remote copy. This keeps large binary payloads out of the Function result.
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
