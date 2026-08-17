from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MODAL_TRELLIS2_",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    port: int = 7863
    dry_run: bool = False
    pipeline: Literal["512", "1024"] = "512"

    # Production GPU selection is intentionally not configurable here.
    # The deployed Modal worker owns one fixed A100-80GB pool with max_containers=1.
    # Alternative GPUs belong in a separate benchmark/experiment path so ordinary
    # requests cannot accidentally create parallel autoscaling pools.


settings = Settings()
