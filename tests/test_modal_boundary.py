from __future__ import annotations

from pathlib import Path


def test_production_gpu_policy_is_single_container() -> None:
    source = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    weights = Path("src/modal_trellis2/modal/weights.py").read_text(encoding="utf-8")
    generator = Path("src/modal_trellis2/modal/generator.py").read_text(encoding="utf-8")

    assert 'PRODUCTION_GPU = "A100-80GB"' in weights
    assert "GPU_MIN_CONTAINERS = 0" in weights
    assert "GPU_MAX_CONTAINERS = 1" in weights
    assert "GPU_BUFFER_CONTAINERS = 0" in weights
    assert "GPU_SCALEDOWN_SECONDS = 10" in weights

    assert "gpu=PRODUCTION_GPU" in source
    assert "min_containers=GPU_MIN_CONTAINERS" in source
    assert "max_containers=GPU_MAX_CONTAINERS" in source
    assert "buffer_containers=GPU_BUFFER_CONTAINERS" in source
    assert "scaledown_window=GPU_SCALEDOWN_SECONDS" in source

    # Production generation must use the deployed class configuration unchanged.
    assert ".with_options(gpu=" not in generator


def test_gpu_worker_is_offline_and_requires_prefetch() -> None:
    source = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    prefetch = Path("src/modal_trellis2/modal/prefetch.py").read_text(encoding="utf-8")

    assert '"HF_HUB_OFFLINE": "1"' in source
    assert '"TRANSFORMERS_OFFLINE": "1"' in source
    assert "require_trellis_bundle" in source
    assert "require_hf_model_dir" in source
    assert "snapshot_download(" in prefetch
    assert "model_volume.commit()" in prefetch


def test_prefetch_does_not_import_gpu_worker() -> None:
    source = Path("src/modal_trellis2/modal/prefetch.py").read_text(encoding="utf-8")
    assert "Trellis2Worker" not in source


def test_cpu_preprocessor_stays_cpu_only() -> None:
    source = Path("src/modal_trellis2/modal/preprocess.py").read_text(encoding="utf-8")
    assert "gpu=" not in source
    assert "scaledown_window=300" in source
