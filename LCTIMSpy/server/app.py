"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..core.errors import CacheError, DatasetError, LCTIMSError, SdkUnavailableError
from ..core.io.cache import Cache
from .events import EventBus
from .jobs import Job, JobManager
from .registry import DatasetRegistry
from .security import SecurityMiddleware
from .settings import ServerSettings

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class AppState:
    settings: ServerSettings
    cache: Cache
    registry: DatasetRegistry
    bus: EventBus
    jobs: JobManager


def get_state(request: Request) -> AppState:
    return request.app.state.lctims


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    cache = Cache(settings.cache_dir)
    registry = DatasetRegistry(settings.state_dir, cache)
    bus = EventBus()

    def on_finished(job: Job) -> None:
        ds = registry.get(job.dataset_id) if job.dataset_id else None
        if ds is None:
            return
        if job.kind == "copy-local" and job.state == "done":
            ds = registry.update(ds.id, local_path=job.result["path"])
        bus.publish("dataset", registry.describe(ds))

    jobs = JobManager(bus, settings.max_concurrent_jobs, on_finished, settings.cancel_grace_s,
                      control_dir=settings.state_dir / "jobs")
    state = AppState(settings, cache, registry, bus, jobs)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cache.sweep_partials()
        yield
        jobs.shutdown()

    app = FastAPI(title="LCTIMSpy", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.lctims = state
    app.add_middleware(SecurityMiddleware, token=settings.token,
                       allowed_hosts=settings.allowed_hosts)

    @app.exception_handler(LCTIMSError)
    async def lctims_error(request: Request, exc: LCTIMSError):
        status = {SdkUnavailableError: 503, DatasetError: 422, CacheError: 409}.get(type(exc), 500)
        return JSONResponse(status_code=status, content={
            "error": type(exc).__name__, "detail": str(exc),
            "remediation": getattr(exc, "remediation", None)})

    @app.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"error": "ValueError", "detail": str(exc)})

    from .routers import analysis, datasets, system
    from .routers import jobs as jobs_router

    for router in (system.router, datasets.router, jobs_router.router, analysis.router):
        app.include_router(router, prefix="/api")

    if (STATIC_DIR / "index.html").is_file():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    else:
        @app.get("/", include_in_schema=False)
        async def no_frontend():
            return JSONResponse({"message": "LCTIMSpy API is running; the web UI is not built yet.",
                                 "docs": "/api/docs"})

    return app


def not_found(what: str, ident: str) -> HTTPException:
    return HTTPException(404, detail=f"{what} {ident!r} not found")
