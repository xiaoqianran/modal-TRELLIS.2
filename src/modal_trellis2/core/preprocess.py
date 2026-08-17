from __future__ import annotations

from PIL import Image

from modal_trellis2.core.image import encode_remote_jpeg, load_image, resize_for_model

# Official TRELLIS.2 preprocess uses alpha > 0.8 * 255 to find the subject.
_ALPHA_CUTOFF = 204


def has_useful_alpha(image: Image.Image) -> bool:
    """True when the image already has a real alpha matte (skip remote background removal)."""
    if image.mode != "RGBA":
        return False
    lo, _hi = image.getchannel("A").getextrema()
    return lo < 255


def crop_to_foreground(image: Image.Image) -> Image.Image:
    """CPU copy of official preprocess_image after the background-removal step."""
    image = resize_for_model(image.convert("RGBA"))
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
    """Return inline-safe JPEG bytes and whether Modal CPU still needs rembg.

    Every remote image call stays below the transport ceiling before `.remote()`.
    Images with an existing alpha matte are also cropped/composited locally.
    """
    image = resize_for_model(load_image(image_bytes))
    if has_useful_alpha(image):
        return encode_remote_jpeg(crop_to_foreground(image)), False
    return encode_remote_jpeg(image), True
