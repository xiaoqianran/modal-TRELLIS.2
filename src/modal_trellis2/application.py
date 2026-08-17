from __future__ import annotations

from modal_trellis2.core.config import Settings, load_settings
from modal_trellis2.core.generator import ImageTo3DGenerator
from modal_trellis2.core.mock import MockGenerator
from modal_trellis2.core.service import GenerateService


def build_service(
    settings: Settings | None = None,
    *,
    dry_run: bool | None = None,
) -> GenerateService:
    """Composition root: choose the concrete generator without leaking Modal into core/."""
    active_settings = settings or load_settings()
    use_mock = active_settings.dry_run if dry_run is None else dry_run
    if use_mock:
        generator: ImageTo3DGenerator = MockGenerator()
    else:
        from modal_trellis2.modal.generator import ModalTrellis2Generator

        generator = ModalTrellis2Generator()
    return GenerateService(active_settings, generator=generator, dry_run=use_mock)


__all__ = ["build_service"]
