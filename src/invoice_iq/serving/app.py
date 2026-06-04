"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from time import perf_counter

from config.settings import Settings
from fastapi import FastAPI, Request, Response

from invoice_iq import __version__
from invoice_iq.serving.deps import AppDeps, build_app_deps
from invoice_iq.serving.routes import router


def create_app(deps: AppDeps | None = None, settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI app.

    Pass `deps` in tests to avoid loading gitignored model artifacts or external
    embedding weights. In production, dependencies are built from settings.
    """
    deps = deps or build_app_deps(settings)
    logging.basicConfig(level=deps.settings.log_level.upper())
    app = FastAPI(
        title="invoice-iq",
        version=__version__,
        description="Local-first invoice intelligence API.",
    )
    app.state.deps = deps

    @app.middleware("http")
    async def record_metrics(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_ms = (perf_counter() - start) * 1000
            deps.metrics.record_request(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                latency_ms=latency_ms,
            )

    app.include_router(router)
    return app
