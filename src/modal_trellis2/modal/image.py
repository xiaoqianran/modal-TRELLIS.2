from __future__ import annotations

import modal

from modal_trellis2.modal.weights import TRELLIS2_SOURCE_REVISION

# CPU-only image for Volume prefetch. Do not put the TRELLIS CUDA stack here.
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.34.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# CPU rembg / crop. Torch CPU only — never attach a GPU to this image.
cpu_runtime_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "transformers==4.57.3",
        "pillow==12.0.0",
        "huggingface_hub>=0.30.0",
        "safetensors",
        "timm==1.0.22",
        "kornia==0.8.2",
        "einops",
    )
    .env(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
)

# Official TRELLIS.2 stack: CUDA 12.4 + PyTorch 2.6.0.
# Keep the prebuilt backend that has now completed all three TRELLIS sampling
# stages on the real production run; do not replace a demonstrated working CUDA
# path based only on generic hardware assumptions.
CUDA_VERSION = "12.4.0"
PREBUILT_WHEELS = "https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210"

trellis2_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "build-essential",
        "ninja-build",
        "libjpeg-dev",
    )
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "triton==3.2.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "pillow==12.0.0",
        "imageio==2.37.2",
        "imageio-ffmpeg==0.6.0",
        "tqdm==4.67.1",
        "easydict==1.13",
        "opencv-python-headless==4.12.0.88",
        "trimesh==4.10.1",
        "transformers==4.57.3",
        "zstandard==0.25.0",
        "kornia==0.8.2",
        "timm==1.0.22",
        "huggingface_hub",
        "safetensors",
        (
            "git+https://github.com/EasternJournalist/utils3d.git"
            "@9a4eb15e4021b67b12c460c7057d642626897ec8"
        ),
    )
    .run_commands(
        "cd /root && git clone https://github.com/microsoft/TRELLIS.2.git",
        f"cd /root/TRELLIS.2 && git checkout --detach {TRELLIS2_SOURCE_REVISION}",
        "cd /root/TRELLIS.2 && git submodule update --init --recursive",
        "rm -rf /root/TRELLIS.2/o-voxel",
    )
    .pip_install(
        f"{PREBUILT_WHEELS}/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl",
        f"{PREBUILT_WHEELS}/cumesh-0.0.1-cp310-cp310-linux_x86_64.whl",
        f"{PREBUILT_WHEELS}/flex_gemm-0.0.1-cp310-cp310-linux_x86_64.whl",
        f"{PREBUILT_WHEELS}/o_voxel-0.0.1-cp310-cp310-linux_x86_64.whl",
        f"{PREBUILT_WHEELS}/nvdiffrast-0.4.0-cp310-cp310-linux_x86_64.whl",
        f"{PREBUILT_WHEELS}/nvdiffrec_render-0.0.0-cp310-cp310-linux_x86_64.whl",
    )
    .env(
        {
            "ATTN_BACKEND": "flash_attn_3",
            "SPARSE_ATTN_BACKEND": "flash_attn_3",
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": "/root/TRELLIS.2",
        }
    )
)
