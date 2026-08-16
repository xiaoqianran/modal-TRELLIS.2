from __future__ import annotations

import modal

from modal_trellis2.modal.image import trellis2_image

# Separate app so this never replaces the deployed Trellis2Worker.
app = modal.App("modal-trellis2-gpu-smoke")


@app.function(gpu="A100", image=trellis2_image, timeout=15 * 60)
def gpu_smoke() -> dict[str, object]:
    """Build the CUDA image and see a GPU. Does not load TRELLIS.2 weights."""
    import torch

    return {
        "ok": torch.cuda.is_available(),
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
    }


@app.local_entrypoint()
def main() -> None:
    print(gpu_smoke.remote())
