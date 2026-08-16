from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from modal_trellis2 import __version__
from modal_trellis2.core.config import Settings, load_settings


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class DoctorReport:
    ready: bool
    version: str
    checks: list[Check] = field(default_factory=list)


def run_doctor(settings: Settings | None = None) -> DoctorReport:
    settings = settings or load_settings()
    checks = [
        _writable("data dir", settings.data_dir),
        _writable("jobs dir", settings.jobs_dir),
        _command("codegraph", "codegraph"),
        _optional_dir("vendor/TRELLIS.2", Path("vendor/TRELLIS.2")),
        _optional_dir("vendor/fast-trellis2", Path("vendor/fast-trellis2")),
        _optional_dir("vendor/meshii", Path("vendor/meshii")),
        Check(
            name="dry-run default",
            ok=True,
            detail="on" if settings.dry_run else "off — this will call Modal if a worker is deployed",
        ),
    ]
    ready = all(check.ok for check in checks if check.name in {"data dir", "jobs dir"})
    return DoctorReport(ready=ready, version=__version__, checks=checks)


def _writable(name: str, path: Path) -> Check:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".doctor-write"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return Check(name, True, str(path.resolve()))
    except OSError as exc:
        return Check(name, False, str(exc))


def _command(name: str, binary: str) -> Check:
    found = shutil.which(binary)
    if found:
        return Check(name, True, found)
    return Check(name, False, f"{binary} not on PATH (optional; used to index vendor repos)")


def _optional_dir(name: str, path: Path) -> Check:
    if path.is_dir() and any(path.iterdir()):
        return Check(name, True, str(path.resolve()))
    return Check(name, False, f"missing — run scripts/fetch-upstream.sh (optional local reference)")
