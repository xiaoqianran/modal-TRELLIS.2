from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modal_trellis2.modal.app import app, huggingface_secret
from modal_trellis2.modal.image import cpu_image
from modal_trellis2.modal.model_bundle import inspect_hf_model_dir, inspect_trellis_bundle
from modal_trellis2.modal.volumes import MODEL_DIR, model_volume
from modal_trellis2.modal.weights import (
    BIREFNET_LOCAL,
    BIREFNET_REPO,
    DINOV3_LOCAL,
    DINOV3_REPO,
    DINOV3_URL,
    RMBG_LOCAL,
    RMBG_REPO,
    SS_DEC_NAME,
    SS_DEC_REPO,
    TRELLIS2_MODEL_REVISION,
    TRELLIS2_REPO,
)

# Keep this CPU prefetch module independent from the GPU worker module.
# Importing the GPU worker here would build the CUDA image just to download weights.


def _hf_env() -> None:
    os.environ["HF_HOME"] = MODEL_DIR


def _extra_dir(name: str) -> str:
    return f"{MODEL_DIR}/{name}"


def _extra_report(name: str) -> dict[str, Any]:
    path = Path(MODEL_DIR) / name
    report = inspect_hf_model_dir(path)
    return {"ok": bool(report["ok"]), "path": str(path), "validation": report}


@app.function(
    image=cpu_image,
    volumes={MODEL_DIR: model_volume},
    secrets=[huggingface_secret()],
    timeout=3600,
)
def prefetch_weights() -> dict[str, Any]:
    """CPU download and full validation of every file the offline GPU needs."""
    from huggingface_hub import HfApi, login, snapshot_download
    from huggingface_hub.errors import GatedRepoError

    _hf_env()
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    api = HfApi()
    previous_manifest = _read_manifest()
    revisions: dict[str, str | None] = {TRELLIS2_REPO: TRELLIS2_MODEL_REVISION}

    dest = f"{MODEL_DIR}/trellis2"
    # The production source and primary model bundle are both pinned. The GPU worker is offline.
    snapshot_download(
        repo_id=TRELLIS2_REPO,
        local_dir=dest,
        ignore_patterns=["*.md", "*.txt"],
        token=token,
        revision=TRELLIS2_MODEL_REVISION,
    )

    extras: dict[str, Any] = {}
    for repo, folder in (
        (DINOV3_REPO, DINOV3_LOCAL),
        (BIREFNET_REPO, BIREFNET_LOCAL),
        (RMBG_REPO, RMBG_LOCAL),
    ):
        dest_extra = _extra_dir(folder)
        try:
            revision = _manifest_revision(previous_manifest, repo) or _repo_revision(
                api, repo, token
            )
            revisions[repo] = revision
            snapshot_download(
                repo_id=repo,
                local_dir=dest_extra,
                token=token,
                revision=revision,
            )
            extras[repo] = _extra_report(folder)
        except GatedRepoError:
            report = _extra_report(folder)
            extras[repo] = {
                **report,
                "gated": True,
                "error": (
                    f"HF account cannot read {repo}. "
                    f"Check access at https://huggingface.co/{repo}"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            report = _extra_report(folder)
            extras[repo] = {
                **report,
                "error": f"{type(exc).__name__}: {exc}",
            }

    ss_revision = _manifest_revision(previous_manifest, SS_DEC_REPO) or _repo_revision(
        api, SS_DEC_REPO, token
    )
    revisions[SS_DEC_REPO] = ss_revision
    extras[SS_DEC_REPO] = _pin_sparse_structure_decoder(dest, token, revision=ss_revision)

    pipeline = Path(dest) / "pipeline.json"
    trellis_validation = inspect_trellis_bundle(Path(dest))
    manifest = _write_manifest(
        revisions=revisions,
        pipeline_path=pipeline,
        extras=extras,
        trellis_validation=trellis_validation,
    )
    model_volume.commit()

    dinov3_ok = bool(extras.get(DINOV3_REPO, {}).get("ok"))
    rembg_ok = bool(extras.get(RMBG_REPO, {}).get("ok"))
    birefnet_ok = bool(extras.get(BIREFNET_REPO, {}).get("ok"))
    return {
        "ok": bool(trellis_validation["ok"]) and dinov3_ok and (rembg_ok or birefnet_ok),
        "path": dest,
        "has_pipeline_json": pipeline.is_file(),
        "bytes": _dir_bytes(dest),
        "trellis_validation": trellis_validation,
        "dinov3": extras.get(DINOV3_REPO),
        "birefnet": extras.get(BIREFNET_REPO),
        "rmbg": extras.get(RMBG_REPO),
        "ss_decoder": extras.get(SS_DEC_REPO),
        "dinov3_url": DINOV3_URL,
        "gpu_downloads": False,
        "manifest": manifest,
    }


@app.function(
    image=cpu_image,
    volumes={MODEL_DIR: model_volume},
    timeout=120,
)
def prefetch_status() -> dict[str, Any]:
    """Inspect the complete offline bundle without downloading or starting a GPU."""
    dest = Path(MODEL_DIR) / "trellis2"
    trellis_validation = inspect_trellis_bundle(dest)
    dinov3 = _extra_report(DINOV3_LOCAL)
    birefnet = _extra_report(BIREFNET_LOCAL)
    rmbg = _extra_report(RMBG_LOCAL)
    manifest = _read_manifest()
    return {
        "ok": (
            bool(trellis_validation["ok"])
            and bool(dinov3["ok"])
            and (bool(rmbg["ok"]) or bool(birefnet["ok"]))
        ),
        "path": str(dest),
        "has_pipeline_json": (dest / "pipeline.json").is_file(),
        "bytes": _dir_bytes(str(dest)) if dest.exists() else 0,
        "trellis_validation": trellis_validation,
        "dinov3": dinov3,
        "birefnet": birefnet,
        "rmbg": rmbg,
        "dinov3_path": _extra_dir(DINOV3_LOCAL),
        "birefnet_path": _extra_dir(BIREFNET_LOCAL),
        "dinov3_url": DINOV3_URL,
        "gpu_downloads": False,
        "manifest": manifest,
    }


@app.local_entrypoint()
def main(status: bool = False) -> None:
    print(prefetch_status.remote() if status else prefetch_weights.remote())


def _manifest_revision(manifest: dict[str, Any] | None, repo_id: str) -> str | None:
    if not isinstance(manifest, dict):
        return None
    repos = manifest.get("repos")
    if not isinstance(repos, dict):
        return None
    revision = repos.get(repo_id)
    return revision if isinstance(revision, str) and revision else None


def _repo_revision(api: Any, repo_id: str, token: str | None) -> str | None:
    try:
        info = api.model_info(repo_id=repo_id, token=token)
        return str(info.sha) if info.sha else None
    except Exception:  # noqa: BLE001 - auxiliary revision metadata must not block downloads
        return None


def _write_manifest(
    *,
    revisions: dict[str, str | None],
    pipeline_path: Path,
    extras: dict[str, Any],
    trellis_validation: dict[str, Any],
) -> dict[str, Any]:
    pipeline_sha256 = None
    if pipeline_path.is_file():
        pipeline_sha256 = hashlib.sha256(pipeline_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "repos": revisions,
        "pipeline_json_sha256": pipeline_sha256,
        "ready": {
            "trellis2": bool(trellis_validation["ok"]),
            "dinov3": bool(extras.get(DINOV3_REPO, {}).get("ok")),
            "birefnet": bool(extras.get(BIREFNET_REPO, {}).get("ok")),
            "rmbg": bool(extras.get(RMBG_REPO, {}).get("ok")),
            "ss_decoder": bool(extras.get(SS_DEC_REPO, {}).get("ok")),
        },
        "trellis_validation": trellis_validation,
    }
    path = Path(MODEL_DIR) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _read_manifest() -> dict[str, Any] | None:
    path = Path(MODEL_DIR) / "manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _pin_sparse_structure_decoder(
    dest: str,
    token: str | None,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Official pipeline.json points this decoder at TRELLIS-image-large. Keep it local."""
    import shutil

    from huggingface_hub import hf_hub_download

    ckpts = Path(dest) / "ckpts"
    ckpts.mkdir(parents=True, exist_ok=True)
    try:
        for suffix in (".json", ".safetensors"):
            downloaded = hf_hub_download(
                SS_DEC_REPO,
                f"ckpts/{SS_DEC_NAME}{suffix}",
                token=token,
                revision=revision,
            )
            target = ckpts / f"{SS_DEC_NAME}{suffix}"
            if Path(downloaded).resolve() != target.resolve():
                shutil.copy2(downloaded, target)
        pipe_path = Path(dest) / "pipeline.json"
        spec = json.loads(pipe_path.read_text(encoding="utf-8"))
        spec["args"]["models"]["sparse_structure_decoder"] = f"ckpts/{SS_DEC_NAME}"
        pipe_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    json_target = ckpts / f"{SS_DEC_NAME}.json"
    weights_target = ckpts / f"{SS_DEC_NAME}.safetensors"
    return {
        "ok": json_target.is_file() and weights_target.is_file(),
        "path": str(weights_target),
        "bytes": weights_target.stat().st_size if weights_target.is_file() else 0,
    }


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total
