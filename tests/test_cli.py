from pathlib import Path

from typer.testing import CliRunner

from modal_trellis2.cli.app import app

runner = CliRunner()


def test_cli_generate_dry_run(tmp_path: Path, sample_png: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    out = tmp_path / "mesh.glb"
    result = runner.invoke(
        app,
        ["generate", str(sample_png), "-o", str(out), "--dry-run", "--pipeline", "512"],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert out.read_bytes()[:4] == b"glTF"


def test_cli_doctor(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "data dir" in result.output


def test_cli_gpu_reuse_probe_requires_explicit_cost_confirmation(sample_png: Path) -> None:
    result = runner.invoke(app, ["verify-gpu-reuse", str(sample_png)])
    assert result.exit_code == 2
    assert "Refusing to start GPU" in result.output
