from __future__ import annotations

import sys

from typer.testing import CliRunner

from modal_trellis2.cli.app import app

runner = CliRunner()


def test_prefetch_module_does_not_import_worker() -> None:
    for name in list(sys.modules):
        if name.startswith("modal_trellis2.modal.worker"):
            del sys.modules[name]
    import modal_trellis2.modal.prefetch  # noqa: F401

    assert "modal_trellis2.modal.worker" not in sys.modules
    assert hasattr(modal_trellis2.modal.prefetch, "prefetch_weights")
    assert hasattr(modal_trellis2.modal.prefetch, "prefetch_status")


def test_preprocess_module_does_not_import_worker() -> None:
    for name in list(sys.modules):
        if name.startswith("modal_trellis2.modal.worker"):
            del sys.modules[name]
    import modal_trellis2.modal.preprocess  # noqa: F401

    assert "modal_trellis2.modal.worker" not in sys.modules
    assert hasattr(modal_trellis2.modal.preprocess, "CpuPreprocessor")


def test_weight_repo_constants() -> None:
    from modal_trellis2.modal.weights import (
        BIREFNET_LOCAL,
        BIREFNET_REPO,
        DINOV3_LOCAL,
        DINOV3_REPO,
        DINOV3_URL,
        TRELLIS2_REPO,
    )

    assert TRELLIS2_REPO == "microsoft/TRELLIS.2-4B"
    assert DINOV3_REPO.startswith("facebook/dinov3")
    assert DINOV3_URL.startswith("https://huggingface.co/")
    assert BIREFNET_REPO.startswith("ZhengPeng7/")
    assert DINOV3_LOCAL == "dinov3"
    assert BIREFNET_LOCAL == "birefnet"
    from modal_trellis2.modal.weights import SS_DEC_NAME, SS_DEC_REPO

    assert SS_DEC_REPO == "microsoft/TRELLIS-image-large"
    assert SS_DEC_NAME.startswith("ss_dec_")


def test_cli_prefetch_and_deploy_help() -> None:
    prefetch = runner.invoke(app, ["prefetch", "--help"])
    assert prefetch.exit_code == 0, prefetch.output
    assert "Volume" in prefetch.output
    deploy = runner.invoke(app, ["deploy", "--help"])
    assert deploy.exit_code == 0, deploy.output
    smoke = runner.invoke(app, ["gpu-smoke", "--help"])
    assert smoke.exit_code == 0, smoke.output
    health = runner.invoke(app, ["health", "--help"])
    assert health.exit_code == 0, health.output


def test_official_model_is_the_default() -> None:
    from modal_trellis2.core.config import Settings
    from modal_trellis2.modal.weights import GPU_SCALEDOWN_SECONDS, TRELLIS2_REPO

    assert TRELLIS2_REPO == "microsoft/TRELLIS.2-4B"
    assert Settings.model_fields["dry_run"].default is False
    assert GPU_SCALEDOWN_SECONDS == 10


def test_gpu_worker_releases_in_ten_seconds() -> None:
    import inspect

    from modal_trellis2.modal import worker
    from modal_trellis2.modal.weights import GPU_SCALEDOWN_SECONDS

    source = inspect.getsource(worker)
    assert GPU_SCALEDOWN_SECONDS == 10
    assert "scaledown_window=GPU_SCALEDOWN_SECONDS" in source
    assert "enable_memory_snapshot=True" in source
    assert "huggingface_secret" not in source
    assert "HF_HUB_OFFLINE" in source
    assert "CpuPreprocessor" not in source
    assert "from modal_trellis2.modal.preprocess" not in source
