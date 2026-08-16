from __future__ import annotations

# Shared Volume layout. Prefetch and the GPU worker must agree.
TRELLIS2_REPO = "microsoft/TRELLIS.2-4B"
DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_URL = f"https://huggingface.co/{DINOV3_REPO}"
BIREFNET_REPO = "ZhengPeng7/BiRefNet"

# TRELLIS.2-4B is public. DINOv3 is gated and required by the official
# image conditioner. BiRefNet is used to drop backgrounds on RGB inputs.
