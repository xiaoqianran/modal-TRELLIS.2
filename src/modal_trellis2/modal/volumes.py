from __future__ import annotations

import modal

MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"
MODEL_VOLUME_NAME = "modal-trellis2-weights"
OUTPUT_VOLUME_NAME = "modal-trellis2-results"

# Model weights are persistent and shared by CPU prefetch + the GPU worker.
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

# Large generated GLBs must not be returned directly from a block_network GPU
# Function. Modal stores Function outputs >2 MiB in object storage, which requires
# outbound access from the container. Instead write the GLB here, commit, return a
# small path, and let the local client read the Volume through the Modal SDK.
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
