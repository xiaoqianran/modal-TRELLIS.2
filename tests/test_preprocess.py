from io import BytesIO
from pathlib import Path
from random import Random

from PIL import Image

from modal_trellis2.core.image import MAX_REMOTE_IMAGE_BYTES, encode_remote_jpeg
from modal_trellis2.core.preprocess import crop_to_foreground, has_useful_alpha, prepare_image


def test_rgb_needs_cpu_rembg(sample_png: Path) -> None:
    payload, needs_rembg = prepare_image(sample_png.read_bytes())
    assert needs_rembg is True
    assert len(payload) <= MAX_REMOTE_IMAGE_BYTES
    with Image.open(BytesIO(payload)) as prepared:
        assert prepared.format == "JPEG"
        assert prepared.mode == "RGB"


def test_large_rgb_is_resized_before_remote_rembg() -> None:
    image = Image.new("RGB", (6000, 4000), (80, 120, 160))
    source = BytesIO()
    image.save(source, format="JPEG", quality=70)
    payload, needs_rembg = prepare_image(source.getvalue())
    assert needs_rembg is True
    with Image.open(BytesIO(payload)) as prepared:
        assert max(prepared.size) <= 1024
    assert len(payload) <= MAX_REMOTE_IMAGE_BYTES


def test_high_entropy_1024_image_stays_inline_safe() -> None:
    rng = Random(42)
    image = Image.frombytes("RGB", (1024, 1024), rng.randbytes(1024 * 1024 * 3))
    payload = encode_remote_jpeg(image)
    assert 0 < len(payload) <= MAX_REMOTE_IMAGE_BYTES
    with Image.open(BytesIO(payload)) as prepared:
        assert prepared.size == (1024, 1024)
        assert prepared.format == "JPEG"


def test_alpha_is_cropped_locally(tmp_path: Path) -> None:
    image = Image.new("RGBA", (64, 48), (0, 0, 0, 0))
    for x in range(20, 40):
        for y in range(10, 30):
            image.putpixel((x, y), (200, 40, 40, 255))
    path = tmp_path / "cutout.png"
    image.save(path)
    assert has_useful_alpha(Image.open(path)) is True
    payload, needs_rembg = prepare_image(path.read_bytes())
    assert needs_rembg is False
    assert len(payload) <= MAX_REMOTE_IMAGE_BYTES
    with Image.open(BytesIO(payload)) as cropped:
        assert cropped.mode == "RGB"
        assert cropped.format == "JPEG"
        assert max(cropped.size) <= 1024


def test_crop_composites_on_black() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((16, 16), (255, 0, 0, 255))
    out = crop_to_foreground(image)
    assert out.mode == "RGB"
    assert out.getpixel((out.size[0] // 2, out.size[1] // 2))[0] > 0
