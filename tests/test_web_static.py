from pathlib import Path


def test_web_batch_queue_is_serial_and_multi_file() -> None:
    html = Path("src/modal_trellis2/web/static/index.html").read_text(encoding="utf-8")
    script = Path("src/modal_trellis2/web/static/app.js").read_text(encoding="utf-8")
    assert "multiple" in html
    assert 'id="queue-list"' in html
    assert "for (const [index, file] of files.entries())" in script
    assert 'await fetch("/api/generate"' in script
    assert "Promise.all" not in script
    assert "MAX_LIVE_BATCH = 20" in script
