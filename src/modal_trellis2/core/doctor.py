from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

from modal_trellis2.core.config import Settings, load_settings


def _writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".doctor-write"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, str(path.resolve())
    except OSError as exc:
        return False, str(exc)


def run_doctor(settings: Settings | None = None) -> dict[str, Any]:
    """Inspect local prerequisites without importing the Modal runtime or starting GPUs."""
    active = settings or load_settings()
    data_ok, data_detail = _writable(active.data_dir)
    jobs_ok, jobs_detail = _writable(active.jobs_dir)
    modal_available = importlib.util.find_spec("modal") is not None
    git_path = shutil.which("git")
    checks = {
        "data dir": {"ok": data_ok, "detail": data_detail},
        "jobs dir": {"ok": jobs_ok, "detail": jobs_detail},
        "python": {"ok": True, "detail": "running"},
        "modal": {
            "ok": modal_available,
            "detail": "importable" if modal_available else "not installed",
        },
        "git": {"ok": git_path is not None, "detail": git_path or "not on PATH"},
    }
    return {
        # Only local storage is required for doctor readiness; Modal/Git are informational.
        "ok": data_ok and jobs_ok,
        "checks": checks,
        "note": "doctor is local-only; GPU policy lives only under modal/ and is tested separately",
    }
