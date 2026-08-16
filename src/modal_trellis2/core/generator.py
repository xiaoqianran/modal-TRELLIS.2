from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field

from modal_trellis2.core.config import PipelineType


class GenerateRequest(BaseModel):
    job_id: str
    image_bytes: bytes
    filename: str = "input.png"
    seed: int = 42
    pipeline: PipelineType = "512"
    texture_size: int = 1024
    remesh: bool = True
    accelerator: Literal["off", "fast"] = "off"
    gpu: str = "A100-80GB"


class GenerateResult(BaseModel):
    job_id: str
    glb_bytes: bytes | None = None
    filename: str = "mesh.glb"
    latency_ms: float = 0.0
    dry_run: bool = False
    error: str | None = None
    telemetry: dict = Field(default_factory=dict)


class ImageTo3DGenerator(Protocol):
    """One image in, one GLB out. CLI and Web both call this."""

    def generate(self, request: GenerateRequest) -> GenerateResult: ...
