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


def test_production_preserves_runtime_validated_flash_attention_backend() -> None:
    image = Path("src/modal_trellis2/modal/image.py").read_text(encoding="utf-8")
    assert "flash_attn_3-3.0.0b1" in image
    assert '"ATTN_BACKEND": "flash_attn_3"' in image
    assert '"SPARSE_ATTN_BACKEND": "flash_attn_3"' in image


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


def test_large_glb_uses_volume_not_function_return_blob() -> None:
    volumes = Path("src/modal_trellis2/modal/volumes.py").read_text(encoding="utf-8")
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    generator = Path("src/modal_trellis2/modal/generator.py").read_text(encoding="utf-8")

    assert 'OUTPUT_VOLUME_NAME = "modal-trellis2-results"' in volumes
    assert "OUTPUT_DIR: output_volume" in worker
    assert "output_volume.commit()" in worker
    assert '"output_path": relative_output' in worker
    assert '"glb_bytes": payload' not in worker
    assert "output_volume.read_file(output_path)" in generator
    assert "output_volume.remove_file(output_path)" in generator


def test_gpu_vram_is_recorded_for_load_infer_export_and_cleanup() -> None:
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    generator = Path("src/modal_trellis2/modal/generator.py").read_text(encoding="utf-8")

    assert "self.vram_after_load = self._vram_stats()" in worker
    assert "torch.cuda.reset_peak_memory_stats()" in worker
    assert "torch.cuda.max_memory_allocated()" in worker
    assert "torch.cuda.max_memory_reserved()" in worker
    assert '"before_infer": vram_before_infer' in worker
    assert '"after_infer": vram_after_infer' in worker
    assert '"after_export": vram_after_export' in worker
    assert '"after_cleanup": vram_after_cleanup' in worker
    assert '"vram": payload.get("vram")' in generator


def test_live_generation_runs_cpu_bundle_preflight_before_gpu_lookup() -> None:
    generator = Path("src/modal_trellis2/modal/generator.py").read_text(encoding="utf-8")
    preflight = generator.index('modal.Function.from_name(APP_NAME, "prefetch_status")')
    gpu_lookup = generator.index('modal.Cls.from_name(APP_NAME, "Trellis2Worker")')
    assert preflight < gpu_lookup
    assert "CPU preflight failed before GPU launch" in generator


def test_catchable_gpu_init_errors_do_not_escape_enter_hook() -> None:
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    assert "self.init_error = None" in worker
    assert "self.init_error = f\"{type(exc).__name__}: {exc}\"" in worker
    assert "GPU initialization failed: {self.init_error}" in worker


def test_gpu_method_returns_portable_error_metadata() -> None:
    worker = Path("src/modal_trellis2/modal/worker.py").read_text(encoding="utf-8")
    assert '"ok": False' in worker
    assert '"error_type": type(exc).__name__' in worker
    assert '"error": str(exc)' in worker
