from io import BytesIO
from pathlib import Path

from PIL import Image

from modal_trellis2.application import build_service
from modal_trellis2.core.glb import is_glb
from modal_trellis2.core.image import ImageError
from modal_trellis2.core.service import GenerateService


def test_image_in_glb_out(tmp_settings, sample_png: Path) -> None:
    service = build_service(tmp_settings, dry_run=True)
    job = service.generate(sample_png.read_bytes(), filename="sample.png", seed=7, pipeline="512")
    assert job.status == "completed"
    assert job.dry_run is True
    glb = service.store.glb_path(job.id).read_bytes()
    assert is_glb(glb)
    assert job.glb_size_bytes == len(glb)


def test_service_normalizes_large_valid_upload_before_generator(tmp_settings) -> None:
    image = Image.new("RGB", (6000, 4000), (100, 80, 60))
    source = BytesIO()
    image.save(source, format="JPEG", quality=70)

    service = build_service(tmp_settings, dry_run=True)
    job = service.generate(source.getvalue(), filename="large.jpg")
    assert job.status == "completed"
    assert int(job.telemetry["width"]) <= 1024
    assert int(job.telemetry["height"]) <= 1024
    stored = Image.open(service.store.image_path(job.id))
    assert max(stored.size) <= 1024


def test_unexpected_generator_exception_marks_job_failed(tmp_settings, sample_png: Path) -> None:
    class ExplodingGenerator:
        def generate(self, request):  # noqa: ANN001, ANN201
            raise KeyError("unexpected-contract-break")

    service = GenerateService(tmp_settings, generator=ExplodingGenerator(), dry_run=False)
    job = service.generate(sample_png.read_bytes(), filename="sample.png")
    assert job.status == "failed"
    assert "unexpected-contract-break" in (job.error or "")


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
