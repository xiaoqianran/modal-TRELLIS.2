from __future__ import annotations

import modal

APP_NAME = "modal-trellis2"

app = modal.App(APP_NAME)


def huggingface_secret() -> modal.Secret:
    return modal.Secret.from_name("huggingface-secret")


__all__ = [
    "APP_NAME",
    "app",
    "huggingface_secret",
]
