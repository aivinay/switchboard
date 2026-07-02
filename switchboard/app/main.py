from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from switchboard import __version__
from switchboard.app.api import admin, chat, health, personal, ui
from switchboard.app.core.config import Settings, get_settings
from switchboard.app.core.logging import configure_logging
from switchboard.app.services.container import build_container
from switchboard.app.storage.db import create_db_engine, init_db
from switchboard.app.storage.repositories import ContextStore

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    resolved_settings = settings or get_settings()
    engine = create_db_engine(resolved_settings.database_url)
    init_db(engine)
    ContextStore(engine).purge_deleted_sessions(
        before=datetime.now(UTC) - timedelta(seconds=10)
    )

    app = FastAPI(
        title="Switchboard",
        version=__version__,
        description="Privacy-aware, local-first router across CLI coding agents and local LLMs.",
    )
    app.state.settings = resolved_settings
    app.state.engine = engine
    app.state.container = build_container(resolved_settings, engine)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(personal.router)
    app.include_router(admin.router)
    app.include_router(ui.router)
    app.mount(
        "/ui/static",
        StaticFiles(directory=STATIC_DIR),
        name="switchboard-ui-static",
    )

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "product": "Switchboard",
            "status": "ok",
            "message": "Open /docs for the API console or use the switchboard CLI.",
            "links": {
                "docs": "/docs",
                "health": "/health",
                "personal_health": "/personal/health",
                "personal_route": "/personal/route",
            },
        }

    @app.get("/ui", include_in_schema=False)
    async def switchboard_ui() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
