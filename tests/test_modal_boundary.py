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


def test_weight_repo_constants() -> None:
    from modal_trellis2.modal.weights import BIREFNET_REPO, DINOV3_REPO, DINOV3_URL, TRELLIS2_REPO

    assert TRELLIS2_REPO == "microsoft/TRELLIS.2-4B"
    assert DINOV3_REPO.startswith("facebook/dinov3")
    assert DINOV3_URL.startswith("https://huggingface.co/")
    assert BIREFNET_REPO.startswith("ZhengPeng7/")


def test_cli_prefetch_and_deploy_help() -> None:
    prefetch = runner.invoke(app, ["prefetch", "--help"])
    assert prefetch.exit_code == 0, prefetch.output
    assert "Volume" in prefetch.output
    deploy = runner.invoke(app, ["deploy", "--help"])
    assert deploy.exit_code == 0, deploy.output
    smoke = runner.invoke(app, ["gpu-smoke", "--help"])
    assert smoke.exit_code == 0, smoke.output
