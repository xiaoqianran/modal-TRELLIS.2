from __future__ import annotations

from pathlib import Path

from modal_trellis2.core.config import PIPELINES, Settings
from modal_trellis2.core.generator import GenerateRequest


def test_core_contract_has_no_gpu_selector(tmp_path: Path) -> None:
    fields = GenerateRequest.__dataclass_fields__
    assert "gpu" not in fields
    settings = Settings(data_dir=tmp_path)
    assert not hasattr(settings, "default_gpu")


def test_production_pipeline_is_512_only() -> None:
    assert PIPELINES == ("512",)
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    weights = Path("src/modal_trellis2/modal/weights.py").read_text(encoding="utf-8")
    assert 'PRODUCTION_PIPELINES: tuple[str, ...] = ("512",)' in weights
    assert "_require_production_pipeline" in worker


def test_production_worker_has_no_input_concurrency_decorator() -> None:
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    assert "@modal.concurrent" not in worker


def test_gpu_timeout_is_cost_bounded() -> None:
    weights = Path("src/modal_trellis2/modal/weights.py").read_text(encoding="utf-8")
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    assert "GPU_TIMEOUT_SECONDS = 10 * 60" in weights
    assert "timeout=GPU_TIMEOUT_SECONDS" in worker


def test_prefetch_readiness_requires_background_model() -> None:
    source = Path("src/modal_trellis2/modal/prefetch.py").read_text(encoding="utf-8")
    assert "(rembg_ok or birefnet_ok)" in source
    assert 'Path(MODEL_DIR) / "manifest.json"' in source
    assert "inspect_trellis_bundle" in source
    assert "inspect_hf_model_dir" in source


def test_core_package_does_not_import_modal_layer() -> None:
    for path in Path("src/modal_trellis2/core").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from modal_trellis2.modal" not in source, path
        assert "import modal_trellis2.modal" not in source, path


def test_production_gpu_blocks_external_network() -> None:
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    assert "block_network=True" in worker
    assert '"HF_HUB_OFFLINE": "1"' in worker
    assert '"TRANSFORMERS_OFFLINE": "1"' in worker


def test_gpu_worker_revalidates_texture_size() -> None:
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    weights = Path("src/modal_trellis2/modal/weights.py").read_text(encoding="utf-8")
    assert "PRODUCTION_TEXTURE_SIZES" in weights
    assert "_require_production_request" in worker


def test_trellis_source_and_primary_model_revisions_are_pinned() -> None:
    image = Path("src/modal_trellis2/modal/image.py").read_text(encoding="utf-8")
    weights = Path("src/modal_trellis2/modal/weights.py").read_text(encoding="utf-8")
    prefetch = Path("src/modal_trellis2/modal/prefetch.py").read_text(encoding="utf-8")

    assert 'TRELLIS2_SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"' in weights
    assert 'TRELLIS2_MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"' in weights
    assert "git checkout --detach {TRELLIS2_SOURCE_REVISION}" in image
    assert "revision=TRELLIS2_MODEL_REVISION" in prefetch
    assert "git clone --depth 1" not in image


def test_a100_production_uses_flash_attention_2_not_fa3() -> None:
    image = Path("src/modal_trellis2/modal/image.py").read_text(encoding="utf-8")
    assert "flash_attn-2.7.3+cu12torch2.6" in image
    assert '"ATTN_BACKEND": "flash_attn"' in image
    assert '"SPARSE_ATTN_BACKEND": "flash_attn"' in image
    assert "flash_attn_3" not in image


def test_trellis_import_never_runs_in_cpu_memory_snapshot() -> None:
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    assert "enable_memory_snapshot=True" not in worker
    assert not any(line.strip() == "@modal.enter(snap=True)" for line in worker.splitlines())
    assert "@modal.enter()" in worker
    assert 'if not torch.cuda.is_available()' in worker
    assert "from trellis2.pipelines import Trellis2ImageTo3DPipeline" in worker


def test_gpu_worker_never_late_loads_missing_models() -> None:
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    assert "_ensure_models" not in worker
    assert "from trellis2 import models" not in worker
    assert "_require_loaded_models" in worker


def test_generated_glb_has_rpc_safety_guard() -> None:
    weights = Path("src/modal_trellis2/modal/weights.py").read_text(encoding="utf-8")
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    assert "MAX_MODAL_RESULT_BYTES = 90 * 1024 * 1024" in weights
    assert "len(payload) > MAX_MODAL_RESULT_BYTES" in worker
