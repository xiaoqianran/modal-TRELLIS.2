from pathlib import Path

from PIL import Image

from modal_trellis2.core.preprocess import crop_to_foreground, has_useful_alpha, prepare_image


def test_rgb_needs_cpu_rembg(sample_png: Path) -> None:
    png, needs_rembg = prepare_image(sample_png.read_bytes())
    assert needs_rembg is True
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_alpha_is_cropped_locally(tmp_path: Path) -> None:
    image = Image.new("RGBA", (64, 48), (0, 0, 0, 0))
    for x in range(20, 40):
        for y in range(10, 30):
            image.putpixel((x, y), (200, 40, 40, 255))
    path = tmp_path / "cutout.png"
    image.save(path)
    assert has_useful_alpha(Image.open(path)) is True
    png, needs_rembg = prepare_image(path.read_bytes())
    assert needs_rembg is False
    out = tmp_path / "out.png"
    out.write_bytes(png)
    cropped = Image.open(out)
    assert cropped.mode == "RGB"
    assert max(cropped.size) <= 1024


def test_crop_composites_on_black() -> None:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.putpixel((16, 16), (255, 0, 0, 255))
    out = crop_to_foreground(image)
    assert out.mode == "RGB"
    assert out.getpixel((out.size[0] // 2, out.size[1] // 2))[0] > 0

