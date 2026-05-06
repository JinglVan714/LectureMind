"""FastAPI routes."""
from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .auth import require_auth
from .ingest.bilibili import BilibiliIngest
from .pipeline import Pipeline
from .render.renderer import Renderer
from .storage.db import Database

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------- request/response models ----------------


class SummarizeRequest(BaseModel):
    url: str = Field(..., description="B 站视频链接或 BV 号")
    force_refresh: bool = False


class SummarizeResponse(BaseModel):
    job_id: str | None = None
    bv_id: str | None = None
    cached: bool = False
    report_url: str | None = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    bv_id: str | None
    error_msg: str | None = None
    report_url: str | None = None


class SummaryListItem(BaseModel):
    bv_id: str
    title: str | None
    author: str | None
    duration: int | None
    cover_url: str | None
    category: str
    created_at: str
    report_url: str


# ---------------- dependencies ----------------


def _get_db(request: Request) -> Database:
    return request.app.state.db


def _get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


# ---------------- routes ----------------


@router.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, Any]:
    return {"status": "ok"}


@router.post("/api/summarize", response_model=SummarizeResponse, dependencies=[Depends(require_auth)])
async def summarize(
    body: SummarizeRequest,
    bg: BackgroundTasks,
    db: Annotated[Database, Depends(_get_db)],
    pipeline: Annotated[Pipeline, Depends(_get_pipeline)],
) -> SummarizeResponse:
    # Parse BV up-front so we can hit cache before spawning a task
    try:
        bv_id = BilibiliIngest.parse_bv(body.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not body.force_refresh:
        existing = await db.get_summary(bv_id)
        if existing and existing.status == "done":
            return SummarizeResponse(
                bv_id=bv_id,
                cached=True,
                report_url=f"/reports/{bv_id}.html",
            )

    job_id = uuid.uuid4().hex
    await db.create_job(job_id, bv_id)

    async def _progress(p: int, _msg: str) -> None:
        await db.update_job(job_id, progress=p, status="running")

    async def _runner() -> None:
        try:
            await pipeline.run(body.url, force_refresh=body.force_refresh, progress_cb=_progress)
            await db.update_job(job_id, status="done", progress=100, bv_id=bv_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            await db.update_job(job_id, status="failed", error_msg=f"{type(exc).__name__}: {exc}"[:500])

    bg.add_task(_runner)
    return SummarizeResponse(job_id=job_id, bv_id=bv_id, cached=False)


@router.get("/api/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_auth)])
async def get_job(
    job_id: str, db: Annotated[Database, Depends(_get_db)]
) -> JobResponse:
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    report_url = None
    if job.status == "done" and job.bv_id:
        report_url = f"/reports/{job.bv_id}.html"
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        bv_id=job.bv_id,
        error_msg=job.error_msg,
        report_url=report_url,
    )


@router.get("/api/summaries", dependencies=[Depends(require_auth)])
async def list_summaries(
    db: Annotated[Database, Depends(_get_db)],
    limit: int = 50,
    offset: int = 0,
) -> list[SummaryListItem]:
    rows = await db.list_summaries(limit=limit, offset=offset)
    return [
        SummaryListItem(
            bv_id=r.bv_id,
            title=r.title,
            author=r.author,
            duration=r.duration,
            cover_url=r.cover_url,
            category=r.category,
            created_at=r.created_at,
            report_url=f"/reports/{r.bv_id}.html",
        )
        for r in rows
    ]


@router.get("/", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def index_page(
    db: Annotated[Database, Depends(_get_db)],
    request: Request,
) -> HTMLResponse:
    summaries = await db.list_summaries(limit=200, offset=0)
    renderer: Renderer = request.app.state.renderer
    css = renderer.load_inline_css()
    html = renderer.render_index(summaries, css)
    return HTMLResponse(html)
