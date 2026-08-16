from __future__ import annotations

import hashlib
import time

from modal_trellis2.core.generator import GenerateRequest, GenerateResult
from modal_trellis2.core.glb import colored_cube_glb
from modal_trellis2.core.image import average_color, load_image


class MockGenerator:
    """Local stand-in so CLI / Web / tests work without Modal or a GPU."""

    def generate(self, request: GenerateRequest) -> GenerateResult:
        started = time.perf_counter()
        image = load_image(request.image_bytes)
        color = average_color(image)
        digest = hashlib.sha256(request.image_bytes).digest()
        size = 0.55 + (digest[0] / 255.0) * 0.28
        glb = colored_cube_glb(color, size=size)
        latency_ms = (time.perf_counter() - started) * 1000
        return GenerateResult(
            job_id=request.job_id,
            glb_bytes=glb,
            filename="mesh.glb",
            latency_ms=latency_ms,
            dry_run=True,
            telemetry={
                "backend": "mock",
                "pipeline": request.pipeline,
                "seed": request.seed,
                "width": image.width,
                "height": image.height,
                "color": [round(c, 4) for c in color],
                "note": "Placeholder cube tinted from the upload. Swap MockGenerator for ModalTrellis2Generator when GPU is ready.",
            },
        )
