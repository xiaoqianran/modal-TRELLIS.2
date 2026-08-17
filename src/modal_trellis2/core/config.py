from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PipelineType = Literal["512"]
PIPELINES: tuple[PipelineType, ...] = ("512",)
TEXTURE_SIZES: tuple[int, ...] = (256, 512, 1024)
MIN_SEED = 0
MAX_SEED = 2_147_483_647

PIPELINE_HINTS: dict[str, dict[str, object]] = {
    "512": {
        "label": "512³",
        "production": True,
        "note": (
            "Current production contract. Higher-resolution pipelines stay experimental "
            "until separately validated."
        ),
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MODAL_TRELLIS2_",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = 7863
    dry_run: bool = False
    pipeline: PipelineType = "512"
    default_seed: int = 42

    @property
    def default_pipeline(self) -> PipelineType:
        return self.pipeline

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


def load_settings() -> Settings:
    value = Settings()
    value.data_dir.mkdir(parents=True, exist_ok=True)
    value.jobs_dir.mkdir(parents=True, exist_ok=True)
    return value


settings = load_settings()
