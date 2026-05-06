"""FastAPI entrypoint. Wires DB, pipeline, static mounts."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .api import router
from .auth import require_auth
from .config import get_settings
from .copilot.api import router as copilot_router
from .copilot.rag import RAGStore
from .pipeline import Pipeline
from .render.renderer import Renderer
from .storage.db import Database


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _setup_logging(settings.log_level)

    db = Database(settings.db_path)
    await db.init()
    pipeline = Pipeline(db)
    renderer = Renderer()

    rag = RAGStore(settings.db_path)
    try:
        await rag.init()
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "RAGStore init failed (%s); Copilot retrieval disabled", exc
        )

    app.state.db = db
    app.state.pipeline = pipeline
    app.state.renderer = renderer
    app.state.rag = rag
    app.state.settings = settings

    logging.getLogger(__name__).info(
        "LectureMind ready · data_dir=%s · vec=%s · text_model=%s · vl_model=%s · copilot_model=%s · dashscope_base_url=%s",
        settings.data_dir.resolve(),
        getattr(rag, "vec_available", False),
        settings.qwen_text_model,
        settings.qwen_vl_model,
        settings.qwen_copilot_model,
        settings.dashscope_base_url,
    )
    try:
        yield
    finally:
        try:
            await rag.close()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(
    title="LectureMind",
    description="Bilibili → interactive lecture HTML",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(copilot_router)

# Static assets (CSS / JS / fonts).  Served without auth so reports can
# load ``/static/copilot.css`` / ``/static/copilot.js`` before the user
# has signed in to /reports — the SSE ``/api/copilot/ask`` endpoint keeps
# its own Basic Auth dependency.
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# Static-ish file routes (kept under auth) -----------------------------------

@app.get("/reports/{bv_id}.html", dependencies=[Depends(require_auth)])
async def serve_report(bv_id: str) -> Response:
    settings = get_settings()
    path = settings.reports_dir / f"{bv_id}.html"
    if not path.exists():
        return Response(status_code=404, content=f"Report {bv_id} not found.")
    return FileResponse(path, media_type="text/html; charset=utf-8")


@app.get("/keyframes/{bv_id}/{filename}", dependencies=[Depends(require_auth)])
async def serve_keyframe(bv_id: str, filename: str) -> Response:
    settings = get_settings()
    path = settings.keyframes_dir / bv_id / filename
    if not path.exists():
        return Response(status_code=404)
    return FileResponse(path, media_type="image/jpeg")


# Convenience entrypoint for `python -m app.main`
def main() -> None:
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
