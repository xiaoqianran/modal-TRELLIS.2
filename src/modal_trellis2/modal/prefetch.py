from __future__ import annotations

import os
from typing import Any

import modal

from modal_trellis2.modal.app import app, huggingface_secret
from modal_trellis2.modal.image import cpu_image
from modal_trellis2.modal.volumes import MODEL_DIR, model_volume

# This module must not import Trellis2Worker. `modal run -m` would otherwise
# build the CUDA image just to download weights.


@app.function(
    image=cpu_image,
    volumes={MODEL_DIR: model_volume},
    secrets=[huggingface_secret()],
    timeout=3600,
)
def prefetch_weights() -> dict[str, Any]:
    """CPU download of microsoft/TRELLIS.2-4B into the Modal volume."""
    from pathlib import Path

    from huggingface_hub import login, snapshot_download

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    dest = f"{MODEL_DIR}/trellis2"
    snapshot_download(
        repo_id="microsoft/TRELLIS.2-4B",
        local_dir=dest,
        ignore_patterns=["*.md", "*.txt"],
        token=token,
    )
    model_volume.commit()
    pipeline = Path(dest) / "pipeline.json"
    return {
        "ok": pipeline.is_file(),
        "path": dest,
        "has_pipeline_json": pipeline.is_file(),
        "bytes": _dir_bytes(dest),
    }


@app.function(
    image=cpu_image,
    volumes={MODEL_DIR: model_volume},
    timeout=120,
)
def prefetch_status() -> dict[str, Any]:
    """Inspect the Volume without downloading."""
    from pathlib import Path

    dest = Path(MODEL_DIR) / "trellis2"
    pipeline = dest / "pipeline.json"
    return {
        "ok": pipeline.is_file(),
        "path": str(dest),
        "has_pipeline_json": pipeline.is_file(),
        "bytes": _dir_bytes(dest) if dest.exists() else 0,
    }


@app.local_entrypoint()
def main(status: bool = False) -> None:
    print(prefetch_status.remote() if status else prefetch_weights.remote())


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total
