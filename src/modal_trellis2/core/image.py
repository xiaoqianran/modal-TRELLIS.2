from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_MODEL_IMAGE_SIDE = 1024
# Modal stores Function inputs/outputs above 2 MiB in object storage. Keep remote
# image calls comfortably below that boundary so a block_network GPU never needs
# the blob transport just to receive its input.
MAX_REMOTE_IMAGE_BYTES = 1_800_000
ALLOWED_MODES = {"RGB", "RGBA", "L", "P"}
_REMOTE_JPEG_QUALITIES = (92, 88, 84, 80, 76, 70, 64, 56, 48, 40)


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


def resize_for_model(image: Image.Image) -> Image.Image:
    """Match TRELLIS.2's 1024px input ceiling before any remote/RPC boundary."""
    longest = max(image.size)
    if longest <= MAX_MODEL_IMAGE_SIDE:
        return image
    scale = MAX_MODEL_IMAGE_SIDE / longest
    return image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def encode_remote_jpeg(
    image: Image.Image,
    *,
    max_bytes: int = MAX_REMOTE_IMAGE_BYTES,
) -> bytes:
    """Encode model input below Modal's inline payload threshold.

    Remote preprocessing and GPU inference only consume RGB pixels, so a bounded
    JPEG is preferable to lossless PNG for transport. Quality is reduced only as
    much as required to stay under the safety ceiling.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    rgb = resize_for_model(image).convert("RGB")
    last_size = 0
    for quality in _REMOTE_JPEG_QUALITIES:
        buffer = io.BytesIO()
        rgb.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=2,
            optimize=True,
        )
        payload = buffer.getvalue()
        last_size = len(payload)
        if last_size <= max_bytes:
            return payload
    raise ImageError(
        "normalized image could not fit the remote inline payload limit "
        f"({last_size} > {max_bytes} bytes)"
    )


def average_color(image: Image.Image) -> tuple[float, float, float]:
    rgb = image.convert("RGB").resize((1, 1), Image.Resampling.BOX)
    pixel = rgb.getpixel((0, 0))
    if not isinstance(pixel, tuple):
        raise ImageError("could not sample image color")
    return (pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0)


def encode_png(image: Image.Image) -> bytes:
    """Lossless local JobStore representation; never used as a remote transport contract."""
    buffer = io.BytesIO()
    resize_for_model(image).convert("RGBA").save(buffer, format="PNG")
    return buffer.getvalue()
