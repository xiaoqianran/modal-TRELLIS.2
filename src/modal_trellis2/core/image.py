from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_MODES = {"RGB", "RGBA", "L", "P"}


class ImageError(ValueError):
    """Uploaded bytes are not a usable still image."""


def load_image(data: bytes) -> Image.Image:
    if not data:
        raise ImageError("empty upload")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError("image larger than 20MB")
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ImageError("image exceeds 40 megapixels")
        image.load()
    except UnidentifiedImageError as exc:
        raise ImageError("not a readable image") from exc
    except OSError as exc:
        raise ImageError(f"cannot decode image: {exc}") from exc
    if image.format not in {"PNG", "JPEG", "WEBP", "BMP"}:
        raise ImageError(f"unsupported image format: {image.format or 'unknown'}")
    if image.mode not in ALLOWED_MODES:
        image = image.convert("RGBA")
    return image


def average_color(image: Image.Image) -> tuple[float, float, float]:
    rgb = image.convert("RGB").resize((1, 1), Image.Resampling.BOX)
    pixel = rgb.getpixel((0, 0))
    if not isinstance(pixel, tuple):
        raise ImageError("could not sample image color")
    return (pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0)


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGBA").save(buffer, format="PNG")
    return buffer.getvalue()
