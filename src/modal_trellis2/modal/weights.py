from __future__ import annotations

# Shared Volume layout. Prefetch and the GPU worker must agree.
TRELLIS2_REPO = "microsoft/TRELLIS.2-4B"
DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_URL = f"https://huggingface.co/{DINOV3_REPO}"
BIREFNET_REPO = "ZhengPeng7/BiRefNet"
RMBG_REPO = "briaai/RMBG-2.0"
SS_DEC_REPO = "microsoft/TRELLIS-image-large"
SS_DEC_NAME = "ss_dec_conv3d_16l8_fp16"

# GPU must die quickly. Weights stay on the CPU Volume / memory snapshot.
GPU_SCALEDOWN_SECONDS = 10

# 512 is the live default. Skip the 1024 flow weights until a job needs them.
MODELS_512: tuple[str, ...] = (
    "sparse_structure_flow_model",
    "sparse_structure_decoder",
    "shape_slat_flow_model_512",
    "shape_slat_decoder",
    "tex_slat_flow_model_512",
    "tex_slat_decoder",
)
MODELS_1024: tuple[str, ...] = MODELS_512 + (
    "shape_slat_flow_model_1024",
    "tex_slat_flow_model_1024",
)

# Stable Volume folders. Do not rely on huggingface_hub's xet/cache layout.
DINOV3_LOCAL = "dinov3"
BIREFNET_LOCAL = "birefnet"
RMBG_LOCAL = "rmbg"

# TRELLIS.2-4B is public. DINOv3 is gated and required by the official
# image conditioner. BiRefNet is used to drop backgrounds on RGB inputs.
# Downloads happen only on the CPU prefetch image. The GPU is offline.
