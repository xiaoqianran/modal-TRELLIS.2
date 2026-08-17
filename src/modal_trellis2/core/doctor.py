from __future__ import annotations

import importlib.util
import shutil
from typing import Any


def run_doctor() -> dict[str, Any]:
    """Inspect local prerequisites without starting a GPU container."""
    checks = {
        "python": True,
        "modal_importable": importlib.util.find_spec("modal") is not None,
        "git": shutil.which("git") is not None,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "gpu_policy": {
            "production_gpu": "A100-80GB",
            "min_containers": 0,
            "max_containers": 1,
            "buffer_containers": 0,
            "scaledown_window_seconds": 10,
            "request_gpu_override": False,
        },
        "note": "doctor is local-only and never starts the production GPU",
    }
