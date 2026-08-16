from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from modal_trellis2 import __version__
from modal_trellis2.core.config import load_settings
from modal_trellis2.core.doctor import run_doctor
from modal_trellis2.core.service import build_service

app = typer.Typer(
    name="modal-trellis2",
    help="Upload an image. Get a GLB back. Web visualizes it.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()


@app.callback()
def _root(
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def generate(
    image: Path = typer.Argument(..., exists=True, readable=True, help="Input PNG/JPEG/WebP"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write GLB here"),
    seed: int = typer.Option(42, "--seed"),
    pipeline: str = typer.Option("512", "--pipeline"),
    dry_run: bool = typer.Option(False, "--dry-run/--live", help="Mock cube vs official TRELLIS.2-4B"),
) -> None:
    """Image in, official TRELLIS.2 GLB out. Pass --dry-run for a local cube."""
    settings = load_settings()
    service = build_service(settings, dry_run=dry_run)
    job = service.generate(image.read_bytes(), filename=image.name, seed=seed, pipeline=pipeline, dry_run=dry_run)
    if job.status != "completed":
        console.print(f"[red]failed[/red] {job.error}")
        raise typer.Exit(1)
    glb_path = service.store.glb_path(job.id)
    if out:
        out.write_bytes(glb_path.read_bytes())
        glb_path = out
    kind = "dry-run cube" if job.dry_run else "official TRELLIS.2-4B"
    console.print(f"[green]{kind}[/green] {job.id} → {glb_path} ({job.glb_size_bytes} bytes, {job.latency_ms:.0f} ms)")
    timings = (job.telemetry or {}).get("timings")
    if timings:
        console.print(timings)


@app.command()
def jobs() -> None:
    """List local generation jobs."""
    store = build_service(load_settings()).store
    table = Table(title="jobs")
    table.add_column("id")
    table.add_column("status")
    table.add_column("pipeline")
    table.add_column("bytes")
    table.add_column("created")
    for job in store.list_jobs():
        table.add_row(job.id, job.status, job.pipeline, str(job.glb_size_bytes), job.created_at)
    console.print(table)


@app.command()
def doctor() -> None:
    """Check local dirs, optional vendor clones, and CodeGraph."""
    report = run_doctor()
    table = Table(title=f"modal-trellis2 {report.version}")
    table.add_column("check")
    table.add_column("ok")
    table.add_column("detail")
    for check in report.checks:
        table.add_row(check.name, "yes" if check.ok else "no", check.detail)
    console.print(table)
    if not report.ready:
        raise typer.Exit(1)


@app.command()
def health(
    gpu: bool = typer.Option(False, "--gpu", help="Start an A100 to ping Trellis2Worker"),
) -> None:
    """CPU Volume check by default. Pass --gpu only when you want to start an A100."""
    import modal

    from modal_trellis2.modal.app import APP_NAME, app as modal_app
    from modal_trellis2.modal.prefetch import prefetch_status
    from modal_trellis2.modal.weights import TRELLIS2_REPO

    if not gpu:
        with modal.enable_output():
            with modal_app.run():
                payload = prefetch_status.remote()
        console.print(payload)
        if not payload.get("ok"):
            console.print("Volume is missing official weights. Run `modal-trellis2 prefetch`.")
            raise typer.Exit(1)
        console.print(f"[green]cpu[/green] {TRELLIS2_REPO} on Volume, GPU not started")
        return

    try:
        worker = modal.Cls.from_name(APP_NAME, "Trellis2Worker")()
        payload = worker.health.remote()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]official worker not reachable[/red] {exc}")
        console.print("Run `modal-trellis2 deploy` after `modal-trellis2 prefetch`.")
        raise typer.Exit(1) from exc
    console.print(payload)
    if not payload.get("ok"):
        raise typer.Exit(1)
    console.print(f"[green]official gpu[/green] {TRELLIS2_REPO} source={payload.get('source')}")


@app.command()
def prefetch(
    status_only: bool = typer.Option(False, "--status", help="Only inspect the Volume"),
) -> None:
    """Download TRELLIS.2-4B onto the Modal Volume (CPU image only)."""
    import modal

    from modal_trellis2.modal.app import app as modal_app
    from modal_trellis2.modal.prefetch import prefetch_status, prefetch_weights

    if not status_only:
        console.print("prefetching microsoft/TRELLIS.2-4B onto the Modal Volume…")
    with modal.enable_output():
        with modal_app.run():
            payload = prefetch_status.remote() if status_only else prefetch_weights.remote()
    console.print(payload)


@app.command()
def deploy() -> None:
    """Build the CUDA image and deploy Trellis2Worker."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "modal", "deploy", "-m", "modal_trellis2.modal.deploy"]
    console.print(" ".join(cmd))
    raise typer.Exit(subprocess.call(cmd))


@app.command("gpu-smoke")
def gpu_smoke() -> None:
    """Build the TRELLIS CUDA image and ping an A100. Does not load weights."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "modal", "run", "-m", "modal_trellis2.modal.gpu_smoke"]
    console.print(" ".join(cmd))
    raise typer.Exit(subprocess.call(cmd))


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    """Local workbench: upload an image, orbit the GLB."""
    import uvicorn

    settings = load_settings()
    console.print(f"workbench → http://{host}:{port or settings.port}")
    uvicorn.run(
        "modal_trellis2.web.server:app",
        host=host,
        port=port or settings.port,
        reload=False,
    )


def main() -> None:
    app()
