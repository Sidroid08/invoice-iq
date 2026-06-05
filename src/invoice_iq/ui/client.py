"""HTTP client for the invoice-iq FastAPI service used by the Streamlit demo.

Wraps the `/health` and `/agent` endpoints. The client depends on a small
structural `HTTPSession` protocol (not a concrete class), so the demo passes a
real `httpx.Client` while tests pass FastAPI's `TestClient` — both satisfy the
interface, keeping tests hermetic (in-process app, no network, no paid key).
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from invoice_iq.schemas.api import AgentRunResponse

DEFAULT_BASE_URL = "http://localhost:8000"


class HTTPResponse(Protocol):
    """The slice of an HTTP response the client uses."""

    def raise_for_status(self) -> object: ...

    def json(self) -> Any: ...  # noqa: ANN401 - response body is arbitrary JSON


class HTTPSession(Protocol):
    """Structural interface satisfied by both `httpx.Client` and `TestClient`."""

    def get(self, url: str) -> HTTPResponse: ...

    def post(self, url: str, *, files: Any, data: Any) -> HTTPResponse: ...  # noqa: ANN401

    def close(self) -> None: ...


class AgentAPIClient:
    """Thin typed client over the invoice-iq API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        session: HTTPSession | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._owns_session = session is None
        self._session: HTTPSession = session or httpx.Client(base_url=base_url, timeout=timeout)

    def health(self) -> dict[str, Any]:
        """GET /health → parsed JSON (raises on non-2xx)."""
        response = self._session.get("/health")
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def run_agent(
        self, pdf_bytes: bytes, filename: str, question: str | None = None
    ) -> AgentRunResponse:
        """POST a PDF to /agent and return the typed agent result."""
        files = {"file": (filename, pdf_bytes, "application/pdf")}
        data: dict[str, str] = {}
        if question:
            data["question"] = question
        response = self._session.post("/agent", files=files, data=data)
        response.raise_for_status()
        return AgentRunResponse.model_validate(response.json())

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> AgentAPIClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
