from __future__ import annotations

import modal

# Official TRELLIS.2 stack: CUDA 12.4 + PyTorch 2.6.0.
# Native extensions come from JeffreyXiang's Space wheels (same set Meshii used),
# so Modal does not compile flash-attn / o-voxel / nvdiffrast from source.
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
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
    )
    .run_commands(
        "cd /root && git clone --depth 1 --recursive https://github.com/microsoft/TRELLIS.2.git",
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
