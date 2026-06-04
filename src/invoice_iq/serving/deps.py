"""Dependency wiring for the FastAPI app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config.settings import Settings, get_settings

from invoice_iq.agent import AgentDeps
from invoice_iq.classifier import Predictor, load_classifier
from invoice_iq.classifier.train import CHECKPOINT_NAME
from invoice_iq.extraction import RuleBasedExtractor
from invoice_iq.extraction.base import Extractor
from invoice_iq.ingestion.base import OCRProvider
from invoice_iq.ingestion.local_ocr import LocalOCRProvider
from invoice_iq.rag import ChromaVectorStore, RAGPipeline, SentenceTransformerEmbedder
from invoice_iq.rag.embeddings import EmbeddingProvider
from invoice_iq.rag.vector_store import VectorStore
from invoice_iq.serving.monitoring import MetricsCollector


@dataclass
class AppDeps:
    """All service dependencies, injectable for tests and swappable for Phase 7."""

    settings: Settings
    predictor: Predictor
    extractor: Extractor
    rag: RAGPipeline
    ocr_provider: OCRProvider = field(default_factory=LocalOCRProvider)
    metrics: MetricsCollector = field(default_factory=MetricsCollector)

    def to_agent_deps(self) -> AgentDeps:
        """Adapt service dependencies to the Phase 5 agent dependency bundle."""
        return AgentDeps(
            predictor=self.predictor,
            extractor=self.extractor,
            rag=self.rag,
            ocr_provider=self.ocr_provider,
        )


def _require_local(name: str, value: str) -> None:
    if value != "local":
        raise NotImplementedError(f"{name}={value!r} is reserved for the Phase 7 GCP swap-in")


def _build_extractor(settings: Settings) -> Extractor:
    if settings.llm_extraction_ready:
        from invoice_iq.extraction.llm_extractor import LLMExtractor  # noqa: PLC0415

        return LLMExtractor.from_settings(settings)
    return RuleBasedExtractor()


def build_app_deps(
    settings: Settings | None = None,
    *,
    predictor: Predictor | None = None,
    extractor: Extractor | None = None,
    embedder: EmbeddingProvider | None = None,
    store: VectorStore | None = None,
) -> AppDeps:
    """Build local serving dependencies from settings.

    Tests pass a trained tiny predictor plus hermetic RAG components. Production
    loads the trained checkpoint from `MODEL_DIR/classifier.pt`.
    """
    settings = settings or get_settings()
    _require_local("OCR_PROVIDER", settings.ocr_provider)
    _require_local("EMBEDDING_PROVIDER", settings.embedding_provider)
    _require_local("VECTOR_STORE", settings.vector_store)
    if settings.classifier_backend != "local":
        raise NotImplementedError(
            "CLASSIFIER_BACKEND='vertex' is reserved for the Phase 7 Vertex endpoint swap-in"
        )

    if predictor is None:
        checkpoint = Path(settings.model_dir) / CHECKPOINT_NAME
        predictor = load_classifier(checkpoint)
    extractor = extractor or _build_extractor(settings)
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    store = store or ChromaVectorStore(persist_dir=Path(settings.chroma_dir))
    return AppDeps(
        settings=settings,
        predictor=predictor,
        extractor=extractor,
        rag=RAGPipeline(embedder, store),
    )
