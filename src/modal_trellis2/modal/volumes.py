from __future__ import annotations

import modal

MODEL_VOLUME_NAME = "modal-trellis2-weights"
OUTPUT_VOLUME_NAME = "modal-trellis2-results"
MODEL_DIR = "/models"
OUTPUT_DIR = "/outputs"

# Models are durable application state. Generated GLBs use a separate transient
# handoff Volume because a network-blocked GPU cannot return large Modal Function
# payloads through object storage. The local JobStore remains the durable owner.
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
