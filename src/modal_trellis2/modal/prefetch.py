from __future__ import annotations

import os
from typing import Any

import modal

from modal_trellis2.modal.app import app, huggingface_secret
from modal_trellis2.modal.image import cpu_image
from modal_trellis2.modal.volumes import MODEL_DIR, model_volume
from modal_trellis2.modal.weights import (
    BIREFNET_LOCAL,
    BIREFNET_REPO,
    DINOV3_LOCAL,
    DINOV3_REPO,
    DINOV3_URL,
    TRELLIS2_REPO,
)

# This module must not import Trellis2Worker. `modal run -m` would otherwise
# build the CUDA image just to download weights.


def _hf_env() -> None:
    os.environ["HF_HOME"] = MODEL_DIR


def _extra_dir(name: str) -> str:
    return f"{MODEL_DIR}/{name}"


def _extra_ready(name: str) -> bool:
    from pathlib import Path

    return (Path(MODEL_DIR) / name / "config.json").is_file()


@app.function(
    image=cpu_image,
    volumes={MODEL_DIR: model_volume},
    secrets=[huggingface_secret()],
    timeout=3600,
)
def prefetch_weights() -> dict[str, Any]:
    """CPU download of TRELLIS.2-4B plus DINOv3 / BiRefNet into the Volume."""
    from pathlib import Path

    from huggingface_hub import login, snapshot_download
    from huggingface_hub.errors import GatedRepoError

    _hf_env()
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    dest = f"{MODEL_DIR}/trellis2"
    # This is the only Hugging Face download path. The GPU worker is offline.
    snapshot_download(
        repo_id=TRELLIS2_REPO,
        local_dir=dest,
        ignore_patterns=["*.md", "*.txt"],
        token=token,
    )

    extras: dict[str, Any] = {}
    for repo, folder in ((DINOV3_REPO, DINOV3_LOCAL), (BIREFNET_REPO, BIREFNET_LOCAL)):
        dest_extra = _extra_dir(folder)
        try:
            snapshot_download(repo_id=repo, local_dir=dest_extra, token=token)
            extras[repo] = {"ok": _extra_ready(folder), "path": dest_extra}
        except GatedRepoError:
            extras[repo] = {
                "ok": False,
                "gated": True,
                "error": f"HF account cannot read {repo}. Accept the license at {DINOV3_URL}",
            }
        except Exception as exc:  # noqa: BLE001
            extras[repo] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    model_volume.commit()
    pipeline = Path(dest) / "pipeline.json"
    dinov3_ok = bool(extras.get(DINOV3_REPO, {}).get("ok"))
    return {
        "ok": pipeline.is_file() and dinov3_ok,
        "path": dest,
        "has_pipeline_json": pipeline.is_file(),
        "bytes": _dir_bytes(dest),
        "dinov3": extras.get(DINOV3_REPO),
        "birefnet": extras.get(BIREFNET_REPO),
        "dinov3_url": DINOV3_URL,
        "gpu_downloads": False,
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
        "ok": pipeline.is_file() and _extra_ready(DINOV3_LOCAL),
        "path": str(dest),
        "has_pipeline_json": pipeline.is_file(),
        "bytes": _dir_bytes(dest) if dest.exists() else 0,
        "dinov3": _extra_ready(DINOV3_LOCAL),
        "birefnet": _extra_ready(BIREFNET_LOCAL),
        "dinov3_path": _extra_dir(DINOV3_LOCAL),
        "birefnet_path": _extra_dir(BIREFNET_LOCAL),
        "dinov3_url": DINOV3_URL,
        "gpu_downloads": False,
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
