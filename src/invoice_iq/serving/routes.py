"""FastAPI routes for the local invoice-iq pipeline."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from invoice_iq import __version__
from invoice_iq.agent import run_agent
from invoice_iq.extraction.base import ExtractionError
from invoice_iq.ingestion.local_ocr import ingest_pdf
from invoice_iq.schemas.api import (
    AgentRunResponse,
    AskResponse,
    ClassificationResponse,
    ExtractionResponse,
    HealthResponse,
    IngestResponse,
    Prediction,
    PredictionRequest,
    PredictionResponse,
)
from invoice_iq.serving.deps import AppDeps
from invoice_iq.serving.monitoring import MetricsSnapshot

router = APIRouter()

PdfUpload = Annotated[UploadFile, File(description="PDF document to process")]
QuestionForm = Annotated[str, Form(min_length=1, description="Question to answer")]


@dataclass(frozen=True)
class UploadedPDF:
    """A request upload materialized to a temporary PDF path."""

    path: Path
    filename: str
    content_type: str
    size_bytes: int


def get_app_deps(request: Request) -> AppDeps:
    deps = getattr(request.app.state, "deps", None)
    if not isinstance(deps, AppDeps):
        raise RuntimeError("FastAPI app dependencies were not initialized")
    return deps


def _vector_count(deps: AppDeps) -> int:
    return deps.rag.store.count()


def _provider_names(deps: AppDeps) -> dict[str, str]:
    return {
        "ocr": deps.ocr_provider.name,
        "classifier": deps.settings.classifier_backend,
        "extractor": deps.extractor.name,
        "embedding": deps.rag.embedder.name,
        "vector_store": deps.settings.vector_store,
    }


@asynccontextmanager
async def _uploaded_pdf(
    file: UploadFile, deps: AppDeps
) -> AsyncIterator[UploadedPDF]:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="missing filename")
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="only PDF uploads are supported",
        )

    content = await file.read()
    max_bytes = deps.settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"upload exceeds {deps.settings.max_upload_mb} MB limit",
        )
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty upload")

    content_type = file.content_type or "application/pdf"
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(prefix="invoice_iq_", suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            temp_path = Path(tmp.name)
        yield UploadedPDF(
            path=temp_path,
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
        )
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _ingest(upload: UploadedPDF, deps: AppDeps) -> IngestResponse:
    raw, ocr = ingest_pdf(upload.path, deps.ocr_provider)
    raw = raw.model_copy(
        update={
            "filename": upload.filename,
            "content_type": upload.content_type,
            "size_bytes": upload.size_bytes,
        }
    )
    return IngestResponse(raw_document=raw, ocr=ocr)


@router.get("/health", response_model=HealthResponse)
async def health(deps: Annotated[AppDeps, Depends(get_app_deps)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        providers=_provider_names(deps),
        vector_count=_vector_count(deps),
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file: PdfUpload, deps: Annotated[AppDeps, Depends(get_app_deps)]
) -> IngestResponse:
    async with _uploaded_pdf(file, deps) as upload:
        return _ingest(upload, deps)


@router.post("/classify", response_model=ClassificationResponse)
async def classify_endpoint(
    file: PdfUpload, deps: Annotated[AppDeps, Depends(get_app_deps)]
) -> ClassificationResponse:
    async with _uploaded_pdf(file, deps) as upload:
        ingested = _ingest(upload, deps)
    document_type, confidence = deps.predictor.predict_ocr(ingested.ocr)
    deps.metrics.record_prediction(document_type)
    return ClassificationResponse(
        raw_document=ingested.raw_document,
        document_type=document_type,
        confidence=confidence,
        ocr_provider=ingested.ocr.provider,
        page_count=ingested.ocr.page_count,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    payload: PredictionRequest, deps: Annotated[AppDeps, Depends(get_app_deps)]
) -> PredictionResponse:
    predictions: list[Prediction] = []
    for instance in payload.instances:
        document_type, confidence = deps.predictor.predict(instance.text)
        deps.metrics.record_prediction(document_type)
        predictions.append(Prediction(document_type=document_type, confidence=confidence))
    return PredictionResponse(predictions=predictions)


@router.post("/extract", response_model=ExtractionResponse)
async def extract_endpoint(
    file: PdfUpload, deps: Annotated[AppDeps, Depends(get_app_deps)]
) -> ExtractionResponse:
    async with _uploaded_pdf(file, deps) as upload:
        ingested = _ingest(upload, deps)
    try:
        invoice = deps.extractor.extract(ingested.ocr)
    except ExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return ExtractionResponse(
        raw_document=ingested.raw_document,
        invoice=invoice,
        extractor=deps.extractor.name,
    )


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    file: PdfUpload,
    question: QuestionForm,
    deps: Annotated[AppDeps, Depends(get_app_deps)],
) -> AskResponse:
    async with _uploaded_pdf(file, deps) as upload:
        ingested = _ingest(upload, deps)
    try:
        invoice = deps.extractor.extract(ingested.ocr)
    except ExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    deps.rag.index_invoice(invoice, ingested.raw_document.doc_id)
    answer = deps.rag.ask(question, k=deps.settings.rag_top_k)
    return AskResponse(raw_document=ingested.raw_document, invoice=invoice, answer=answer)


@router.post("/agent", response_model=AgentRunResponse)
async def agent_endpoint(
    file: PdfUpload,
    deps: Annotated[AppDeps, Depends(get_app_deps)],
    question: Annotated[str | None, Form(description="Optional question for the RAG step")] = None,
) -> AgentRunResponse:
    async with _uploaded_pdf(file, deps) as upload:
        result = run_agent(deps.to_agent_deps(), upload.path, question=question)
    deps.metrics.record_prediction(result.document_type)
    return AgentRunResponse.model_validate(result.model_dump())


@router.get("/metrics", response_model=MetricsSnapshot)
async def metrics(deps: Annotated[AppDeps, Depends(get_app_deps)]) -> MetricsSnapshot:
    return deps.metrics.snapshot(vector_count=_vector_count(deps))
