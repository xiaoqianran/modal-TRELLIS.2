from __future__ import annotations

from io import BytesIO

from PIL import Image

from modal_trellis2.core.image import load_image

# Official TRELLIS.2 preprocess uses alpha > 0.8 * 255 to find the subject and
# bounds the input to a 1024px longest side before background removal.
_ALPHA_CUTOFF = 204
_MAX_SIDE = 1024


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _resize_to_max_side(image: Image.Image) -> Image.Image:
    longest = max(image.size)
    if longest <= _MAX_SIDE:
        return image
    scale = _MAX_SIDE / longest
    return image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def has_useful_alpha(image: Image.Image) -> bool:
    """True when the image already has a real alpha matte (skip BiRefNet)."""
    if image.mode != "RGBA":
        return False
    lo, _hi = image.getchannel("A").getextrema()
    return lo < 255


def crop_to_foreground(image: Image.Image) -> Image.Image:
    """CPU copy of official preprocess_image after the rembg step."""
    image = _resize_to_max_side(image.convert("RGBA"))
    mask = image.getchannel("A").point(lambda pixel: 255 if pixel > _ALPHA_CUTOFF else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return image.convert("RGB")
    left, top, right, bottom = bbox
    size = max(right - left, bottom - top)
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    half = size / 2
    cropped = image.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
    background = Image.new("RGBA", cropped.size, (0, 0, 0, 255))
    return Image.alpha_composite(background, cropped).convert("RGB")


def prepare_image(image_bytes: bytes) -> tuple[bytes, bool]:
    """Return bounded PNG bytes and whether Modal CPU still needs background removal.

    Resize happens before any remote call. This matches the official 1024px input
    ceiling and prevents a compressed high-resolution upload from expanding into an
    oversized lossless PNG on the Modal RPC boundary.
    """
    image = _resize_to_max_side(load_image(image_bytes))
    if has_useful_alpha(image):
        return _png_bytes(crop_to_foreground(image)), False
    return _png_bytes(image.convert("RGB")), True
