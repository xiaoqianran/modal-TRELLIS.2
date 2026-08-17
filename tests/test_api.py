from pathlib import Path


def test_health(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_meta_names_official_model(client) -> None:
    response = client.get("/api/meta")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "microsoft/TRELLIS.2-4B"


def test_generate_returns_glb(client, sample_png: Path) -> None:
    response = client.post(
        "/api/generate",
        files={"image": ("sample.png", sample_png.read_bytes(), "image/png")},
        data={"seed": "3", "pipeline": "512", "dry_run": "true"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["asset_url"].endswith(".glb")
    glb = client.get(body["asset_url"])
    assert glb.status_code == 200
    assert glb.content[:4] == b"glTF"
    assert glb.headers["content-type"] == "model/gltf-binary"


def test_generate_rejects_text(client) -> None:
    response = client.post(
        "/api/generate",
        files={"image": ("notes.txt", b"not-an-image", "text/plain")},
        data={"dry_run": "true"},
    )
    assert response.status_code == 400
