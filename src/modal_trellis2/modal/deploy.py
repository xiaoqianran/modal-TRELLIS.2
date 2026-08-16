"""Single `modal deploy` target.

The GPU worker module must not import CpuPreprocessor — that file is
CPU-only and would pull the rembg image into the CUDA container import
graph. Import both here so one deploy registers the whole app.
"""

from modal_trellis2.modal.app import app
from modal_trellis2.modal.prefetch import prefetch_status, prefetch_weights
from modal_trellis2.modal.preprocess import CpuPreprocessor
from modal_trellis2.modal.worker import Trellis2Worker

__all__ = [
    "CpuPreprocessor",
    "Trellis2Worker",
    "app",
    "prefetch_status",
    "prefetch_weights",
]
