from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from modal_trellis2 import __version__
from modal_trellis2.application import build_service
from modal_trellis2.core.config import load_settings

app = typer.Typer(
    name="modal-trellis2",
    help="Upload an image. Get a GLB back. Local workbench, fixed Modal GPU pool.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()


@app.callback()
def _root(version: bool = typer.Option(False, "--version", help="Show version and exit")) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def doctor() -> None:
    """Inspect local prerequisites without starting a GPU."""
    from modal_trellis2.core.doctor import run_doctor

    result = run_doctor(load_settings())
    console.print_json(json.dumps(result))
    if not result.get("ok"):
        raise typer.Exit(1)


@app.command()
def prefetch(
    status: bool = typer.Option(False, "--status", help="Only inspect the Volume"),
) -> None:
    """CPU-only model download/status. This intentionally works before production deploy."""
    try:
        import modal

        from modal_trellis2.modal.app import app as modal_app
        from modal_trellis2.modal.prefetch import prefetch_status, prefetch_weights
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Modal import failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        with modal.enable_output():
            with modal_app.run():
                result = prefetch_status.remote() if status else prefetch_weights.remote()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Prefetch failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(json.dumps(result))
    if not result.get("ok"):
        raise typer.Exit(1)


@app.command()
def deploy() -> None:
    """Deploy CPU preprocessor + the fixed one-container production GPU worker."""
    cmd = [sys.executable, "-m", "modal", "deploy", "-m", "modal_trellis2.modal.deploy"]
    console.print(" ".join(cmd))
    raise typer.Exit(subprocess.call(cmd))


@app.command()
def health(
    gpu: bool = typer.Option(False, "--gpu", help="Explicitly start production GPU health probe"),
) -> None:
    """CPU Volume status by default. --gpu is the only health mode that starts A100."""
    try:
        import modal

        from modal_trellis2.modal.app import APP_NAME
        from modal_trellis2.modal.app import app as modal_app
        from modal_trellis2.modal.prefetch import prefetch_status
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Modal import failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        if gpu:
            target = modal.Cls.from_name(APP_NAME, "Trellis2Worker")()
            result = target.health.remote()
        else:
            with modal.enable_output():
                with modal_app.run():
                    result = prefetch_status.remote()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Health check failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print_json(json.dumps(result))
    if not result.get("ok"):
        raise typer.Exit(1)


@app.command()
def generate(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "--out", "-o")] = Path("output.glb"),
    pipeline: Annotated[str, typer.Option("--pipeline")] = "512",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    texture_size: Annotated[int, typer.Option("--texture-size")] = 1024,
    remesh: Annotated[bool, typer.Option("--remesh/--no-remesh")] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run/--live")] = False,
) -> None:
    """Generate one GLB. Production GPU is fixed and cannot be overridden per request."""
    settings = load_settings()
    service = build_service(settings, dry_run=dry_run)
    try:
        result = service.generate_path(
            image,
            output,
            pipeline=pipeline,
            seed=seed,
            texture_size=texture_size,
            remesh=remesh,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if result.status != "completed":
        console.print(f"[red]{result.error or 'generation failed'}[/red]")
        raise typer.Exit(1)
    console.print_json(json.dumps(result.public_dict()))


@app.command("jobs")
def jobs_cmd(limit: Annotated[int, typer.Option("--limit")] = 20) -> None:
    service = build_service(load_settings(), dry_run=True)
    console.print_json(
        json.dumps([job.public_dict() for job in service.store.list_jobs(limit=limit)])
    )


@app.command("verify-gpu-reuse")
def verify_gpu_reuse(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    count: Annotated[int, typer.Option("--count", min=2, max=10)] = 3,
    check_scale_down: Annotated[bool, typer.Option("--check-scale-down")] = False,
    idle_seconds: Annotated[int, typer.Option("--idle-seconds", min=12, max=120)] = 15,
    confirm_cost: Annotated[bool, typer.Option("--confirm-cost")] = False,
) -> None:
    """Explicit live acceptance test for one-container reuse. This starts billable GPU work."""
    if not confirm_cost:
        console.print(
            "[red]Refusing to start GPU.[/red] Re-run with --confirm-cost after reviewing cost."
        )
        raise typer.Exit(2)

    import time

    settings = load_settings()
    service = build_service(settings, dry_run=False)
    payload = image.read_bytes()
    instance_ids: list[str] = []

    for index in range(count):
        job = service.generate(
            payload,
            filename=image.name,
            pipeline="512",
            seed=settings.default_seed + index,
        )
        if job.status != "completed":
            console.print(f"[red]generation failed[/red] {job.error}")
            raise typer.Exit(1)
        instance_id = str((job.telemetry or {}).get("container_instance_id") or "")
        if not instance_id:
            console.print(
                "[red]container_instance_id missing.[/red] Redeploy this repository before testing."
            )
            raise typer.Exit(1)
        instance_ids.append(instance_id)
        console.print(f"job {index + 1}/{count}: container={instance_id}")

    reused = len(set(instance_ids)) == 1
    console.print({"sequential_reuse": reused, "container_ids": instance_ids})
    if not reused:
        raise typer.Exit(1)

    if not check_scale_down:
        return

    console.print(f"waiting {idle_seconds}s before one explicit post-idle probe...")
    time.sleep(idle_seconds)
    job = service.generate(payload, filename=image.name, pipeline="512", seed=999_999)
    if job.status != "completed":
        console.print(f"[red]post-idle generation failed[/red] {job.error}")
        raise typer.Exit(1)
    after_id = str((job.telemetry or {}).get("container_instance_id") or "")
    recreated = bool(after_id) and after_id != instance_ids[-1]
    console.print(
        {
            "post_idle_recreated": recreated,
            "before": instance_ids[-1],
            "after": after_id,
            "idle_seconds": idle_seconds,
        }
    )
    if not recreated:
        raise typer.Exit(1)


@app.command()
def web(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int | None, typer.Option("--port")] = None,
) -> None:
    """Run the local FastAPI workbench. Expensive inference stays on Modal."""
    import uvicorn

    settings = load_settings()
    active_port = port or settings.port
    console.print(f"workbench → http://{host}:{active_port}")
    uvicorn.run("modal_trellis2.web.server:app", host=host, port=active_port, reload=False)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
