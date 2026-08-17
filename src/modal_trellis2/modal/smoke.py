from __future__ import annotations

import modal

from modal_trellis2.modal.app import huggingface_secret

app = modal.App("modal-trellis2-smoke")


@app.function(secrets=[huggingface_secret()])
def ping() -> dict[str, object]:
    """Cheap auth + secret check. No GPU, no TRELLIS image."""
    token = os_env_present("HF_TOKEN")
    return {
        "ok": True,
        "app": "modal-trellis2-smoke",
        "hf_token": token,
    }


def os_env_present(name: str) -> bool:
    import os

    return bool(os.environ.get(name))


@app.local_entrypoint()
def main() -> None:
    print(ping.remote())
