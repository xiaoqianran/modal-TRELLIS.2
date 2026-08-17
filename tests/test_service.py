from pathlib import Path

from modal_trellis2.application import build_service
from modal_trellis2.core.glb import is_glb
from modal_trellis2.core.image import ImageError


def test_image_in_glb_out(tmp_settings, sample_png: Path) -> None:
    service = build_service(tmp_settings, dry_run=True)
    job = service.generate(sample_png.read_bytes(), filename="sample.png", seed=7, pipeline="512")
    assert job.status == "completed"
    assert job.dry_run is True
    glb = service.store.glb_path(job.id).read_bytes()
    assert is_glb(glb)
    assert job.glb_size_bytes == len(glb)


def test_rejects_non_image(tmp_settings) -> None:
    service = build_service(tmp_settings, dry_run=True)
    try:
        service.generate(b"hello", filename="notes.txt")
    except ImageError:
        return
    raise AssertionError("expected ImageError")


def test_rejects_unknown_pipeline(tmp_settings, sample_png: Path) -> None:
    service = build_service(tmp_settings, dry_run=True)
    try:
        service.generate(sample_png.read_bytes(), pipeline="4096")
    except ValueError as exc:
        assert "pipeline" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_rejects_unsafe_texture_size(tmp_settings, sample_png: Path) -> None:
    service = build_service(tmp_settings, dry_run=True)
    try:
        service.generate(sample_png.read_bytes(), texture_size=4096)
    except ValueError as exc:
        assert "texture_size" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_rejects_negative_seed(tmp_settings, sample_png: Path) -> None:
    service = build_service(tmp_settings, dry_run=True)
    try:
        service.generate(sample_png.read_bytes(), seed=-1)
    except ValueError as exc:
        assert "seed" in str(exc)
        return
    raise AssertionError("expected ValueError")
