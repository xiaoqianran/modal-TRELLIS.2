from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from modal_trellis2.core.config import Settings
from modal_trellis2.core.service import build_service
from modal_trellis2.web.api import configure
from modal_trellis2.web.server import create_app


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=tmp_path / "data", dry_run=True, port=7863)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    return settings


@pytest.fixture
def sample_png(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    Image.new("RGB", (64, 48), (197, 106, 58)).save(path)
    return path


@pytest.fixture
def client(tmp_settings: Settings) -> TestClient:
    service = build_service(tmp_settings, dry_run=True)
    configure(tmp_settings, service)
    return TestClient(create_app())
