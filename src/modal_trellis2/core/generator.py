from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from modal_trellis2.core.config import PipelineType


@dataclass(slots=True)
class GenerateRequest:
    job_id: str
    image_bytes: bytes
    pipeline: PipelineType = "512"
    seed: int = 42
    texture_size: int = 1024
    remesh: bool = True


@dataclass(slots=True)
class GenerateResult:
    job_id: str
    glb_bytes: bytes | None = None
    filename: str = "mesh.glb"
    error: str | None = None
    latency_ms: float = 0.0
    dry_run: bool = False
    telemetry: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.glb_bytes is not None


class ImageTo3DGenerator(Protocol):
    """One normalized image in, one GLB out. No GPU choice belongs in this contract."""

    def generate(self, request: GenerateRequest) -> GenerateResult: ...
