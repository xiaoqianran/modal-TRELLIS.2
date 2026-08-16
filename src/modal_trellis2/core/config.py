from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PipelineType = Literal["512", "1024", "1024_cascade", "1536_cascade"]

PIPELINES: tuple[PipelineType, ...] = ("512", "1024", "1024_cascade", "1536_cascade")

# Official H100 numbers from microsoft/TRELLIS.2. Modal A100-80GB is slower.
PIPELINE_HINTS: dict[str, dict[str, object]] = {
    "512": {
        "label": "512³",
        "min_vram_gb": 16,
        "gpu": "A100",
        "seconds": (3, 12),
        "note": "First live GPU run. Cheap enough to debug the contract.",
    },
    "1024": {
        "label": "1024³",
        "min_vram_gb": 24,
        "gpu": "A100-80GB",
        "seconds": (12, 30),
        "note": "Single-stage 1024. Use after 512 works.",
    },
    "1024_cascade": {
        "label": "1024 cascade",
        "min_vram_gb": 24,
        "gpu": "A100-80GB",
        "seconds": (15, 40),
        "note": "Official default quality. Not the first GPU call.",
    },
    "1536_cascade": {
        "label": "1536 cascade",
        "min_vram_gb": 40,
        "gpu": "H100",
        "seconds": (50, 90),
        "note": "Highest quality. Leave until the web loop is boring.",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MODAL_TRELLIS2_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    port: int = 7863
    host: str = "127.0.0.1"
    default_gpu: str = "A100-80GB"
    default_pipeline: PipelineType = "512"
    default_seed: int = 42
    dry_run: bool = False
    hf_token: str | None = None

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


def load_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    return settings
