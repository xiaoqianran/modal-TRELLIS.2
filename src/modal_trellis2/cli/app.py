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
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Mock cube vs Modal GPU"),
) -> None:
    """Image in, GLB file out."""
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
    kind = "dry-run cube" if job.dry_run else "TRELLIS.2"
    console.print(f"[green]{kind}[/green] {job.id} → {glb_path} ({job.glb_size_bytes} bytes, {job.latency_ms:.0f} ms)")


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
