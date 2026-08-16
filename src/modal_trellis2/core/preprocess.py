from __future__ import annotations

from io import BytesIO

from PIL import Image

from modal_trellis2.core.image import load_image


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()

# Official TRELLIS.2 preprocess uses alpha > 0.8 * 255 to find the subject.
_ALPHA_CUTOFF = 204
_MAX_SIDE = 1024


def has_useful_alpha(image: Image.Image) -> bool:
    """True when the image already has a real alpha matte (skip BiRefNet)."""
    if image.mode != "RGBA":
        return False
    lo, _hi = image.getchannel("A").getextrema()
    return lo < 255


def crop_to_foreground(image: Image.Image) -> Image.Image:
    """CPU copy of official preprocess_image after the rembg step."""
    image = image.convert("RGBA")
    longest = max(image.size)
    if longest > _MAX_SIDE:
        scale = _MAX_SIDE / longest
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
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
    """Return PNG bytes and whether Modal CPU still needs to run BiRefNet.

    Local CPU handles resize/crop when the upload already has alpha.
    RGB photos go to the CPU rembg worker; the GPU never loads BiRefNet.
    """
    image = load_image(image_bytes)
    if has_useful_alpha(image):
        return _png_bytes(crop_to_foreground(image)), False
    return _png_bytes(image.convert("RGB")), True
