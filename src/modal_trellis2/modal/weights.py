from __future__ import annotations

# Shared Volume layout. Prefetch and the GPU worker must agree.
TRELLIS2_REPO = "microsoft/TRELLIS.2-4B"
TRELLIS2_SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
# Keep model metadata/weights aligned with the source revision above. A floating
# Hugging Face HEAD can silently change pipeline.json underneath pinned code.
TRELLIS2_MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_URL = f"https://huggingface.co/{DINOV3_REPO}"
BIREFNET_REPO = "ZhengPeng7/BiRefNet"
RMBG_REPO = "briaai/RMBG-2.0"
SS_DEC_REPO = "microsoft/TRELLIS-image-large"
SS_DEC_NAME = "ss_dec_conv3d_16l8_fp16"

# Production cost policy.
# Keep a single fixed GPU pool, queue bursts onto it, then scale to zero quickly.
PRODUCTION_GPU = "A100-80GB"
GPU_SCALEDOWN_SECONDS = 10
GPU_TIMEOUT_SECONDS = 10 * 60
GPU_MIN_CONTAINERS = 0
GPU_MAX_CONTAINERS = 1
GPU_BUFFER_CONTAINERS = 0
PRODUCTION_PIPELINES: tuple[str, ...] = ("512",)
PRODUCTION_TEXTURE_SIZES: tuple[int, ...] = (256, 512, 1024)

# Modal's RPC transport has a finite payload ceiling. Keep a margin so a large
# generated GLB fails with a readable worker error rather than a transport error.
MAX_MODAL_RESULT_BYTES = 90 * 1024 * 1024

# Production currently loads only the validated 512 model set.
MODELS_512: tuple[str, ...] = (
    "sparse_structure_flow_model",
    "sparse_structure_decoder",
    "shape_slat_flow_model_512",
    "shape_slat_decoder",
    "tex_slat_flow_model_512",
    "tex_slat_decoder",
)

# Stable Volume folders. Do not rely on huggingface_hub's xet/cache layout.
DINOV3_LOCAL = "dinov3"
BIREFNET_LOCAL = "birefnet"
RMBG_LOCAL = "rmbg"

# TRELLIS.2-4B is public. DINOv3 is gated and required by the official
# image conditioner. BiRefNet is used to drop backgrounds on RGB inputs.
# Downloads happen only on the CPU prefetch image. The GPU is offline.
