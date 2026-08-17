from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from modal_trellis2.core.config import settings
from modal_trellis2.core.jobs import JobStore
from modal_trellis2.core.service import GenerateService

app = typer.Typer(no_args_is_help=True)
console = Console()


def _service(dry_run: bool | None = None) -> GenerateService:
    return GenerateService(
        data_dir=settings.data_dir,
        dry_run=settings.dry_run if dry_run is None else dry_run,
    )


@app.command()
def doctor() -> None:
    """Inspect local prerequisites without starting a GPU."""
    from modal_trellis2.core.doctor import run_doctor

    result = run_doctor()
    console.print_json(json.dumps(result))
    if not result.get("ok"):
        raise typer.Exit(1)


@app.command()
def prefetch(status: bool = typer.Option(False, "--status")) -> None:
    """Download model weights on CPU or inspect the shared Volume."""
    try:
        import modal

        from modal_trellis2.modal.app import APP_NAME
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Modal import failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        fn_name = "prefetch_status" if status else "prefetch_weights"
        fn = modal.Function.from_name(APP_NAME, fn_name)
        result = fn.remote()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Prefetch failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(json.dumps(result))
    if not result.get("ok"):
        raise typer.Exit(1)


@app.command()
def deploy() -> None:
    """Deploy the Modal app. Does not keep a GPU warm."""
    from modal_trellis2.modal.deploy import deploy_app

    deploy_app()


@app.command()
def health(gpu: bool = typer.Option(False, "--gpu", help="Also start the production A100 health probe.")) -> None:
    """Check the CPU Volume by default; --gpu explicitly starts the one production GPU container."""
    try:
        import modal

        from modal_trellis2.modal.app import APP_NAME
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Modal import failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        if gpu:
            target = modal.Cls.from_name(APP_NAME, "Trellis2Worker")()
            result = target.health.remote()
        else:
            target = modal.Function.from_name(APP_NAME, "prefetch_status")
            result = target.remote()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Health check failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(json.dumps(result))
    if not result.get("ok"):
        raise typer.Exit(1)


@app.command()
def generate(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("output.glb"),
    pipeline: Annotated[str, typer.Option("--pipeline")] = "512",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    texture_size: Annotated[int, typer.Option("--texture-size")] = 1024,
    remesh: Annotated[bool, typer.Option("--remesh/--no-remesh")] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Generate one GLB. Production GPU type is fixed and cannot be overridden per request."""
    service = _service(dry_run=dry_run)
    result = service.generate_path(
        image,
        output,
        pipeline=pipeline,
        seed=seed,
        texture_size=texture_size,
        remesh=remesh,
    )
    if result.error:
        console.print(f"[red]{result.error}[/red]")
        raise typer.Exit(1)
    console.print_json(json.dumps(result.model_dump(mode="json")))


@app.command()
def web(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = settings.port,
) -> None:
    """Run the local FastAPI workbench. GPU stays remote on Modal."""
    import uvicorn

    uvicorn.run("modal_trellis2.web.server:app", host=host, port=port, reload=False)


@app.command("jobs")
def jobs_cmd(limit: Annotated[int, typer.Option("--limit")] = 20) -> None:
    store = JobStore(settings.data_dir)
    console.print_json(json.dumps([job.model_dump(mode="json") for job in store.list(limit=limit)]))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
