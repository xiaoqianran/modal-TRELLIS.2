from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from modal_trellis2.web.api import router

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    application = FastAPI(title="modal-TRELLIS.2", docs_url="/api/docs")
    application.include_router(router)

    @application.get("/")
    def index() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            raise HTTPException(500, "web/static/index.html is missing")
        return FileResponse(page)

    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return application


app = create_app()
