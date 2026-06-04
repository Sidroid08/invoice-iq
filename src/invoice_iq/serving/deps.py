"""Dependency wiring for the FastAPI app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config.settings import Settings, get_settings

from invoice_iq.agent import AgentDeps
from invoice_iq.classifier import DocumentTypePredictor, load_classifier
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
    predictor: DocumentTypePredictor
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


def _build_extractor(settings: Settings) -> Extractor:
    if settings.llm_extraction_ready:
        from invoice_iq.extraction.llm_extractor import LLMExtractor  # noqa: PLC0415

        return LLMExtractor.from_settings(settings)
    return RuleBasedExtractor()


def _build_ocr_provider(settings: Settings) -> OCRProvider:
    if settings.ocr_provider == "local":
        return LocalOCRProvider()
    from invoice_iq.ingestion.docai import DocAIOCRProvider  # noqa: PLC0415

    return DocAIOCRProvider.from_settings(settings)


def _build_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "local":
        return SentenceTransformerEmbedder(settings.embedding_model)
    from invoice_iq.vertex import VertexEmbedder  # noqa: PLC0415

    return VertexEmbedder.from_settings(settings)


def _build_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "local":
        return ChromaVectorStore(persist_dir=Path(settings.chroma_dir))
    raise NotImplementedError(
        "VECTOR_STORE='gcp' requires a Vertex Vector Search index and is gated "
        "behind explicit Phase 7 cost approval. Leave VECTOR_STORE=local to use Chroma."
    )


def _build_predictor(settings: Settings) -> DocumentTypePredictor:
    if settings.classifier_backend == "local":
        checkpoint = Path(settings.model_dir) / CHECKPOINT_NAME
        return load_classifier(checkpoint)
    from invoice_iq.vertex import VertexClassifierClient  # noqa: PLC0415

    return VertexClassifierClient.from_settings(settings)


def build_app_deps(
    settings: Settings | None = None,
    *,
    predictor: DocumentTypePredictor | None = None,
    extractor: Extractor | None = None,
    embedder: EmbeddingProvider | None = None,
    store: VectorStore | None = None,
) -> AppDeps:
    """Build local serving dependencies from settings.

    Tests pass a trained tiny predictor plus hermetic RAG components. Production
    loads the trained checkpoint from `MODEL_DIR/classifier.pt`.
    """
    settings = settings or get_settings()
    ocr_provider = _build_ocr_provider(settings)
    predictor = predictor or _build_predictor(settings)
    extractor = extractor or _build_extractor(settings)
    embedder = embedder or _build_embedder(settings)
    store = store or _build_store(settings)
    return AppDeps(
        settings=settings,
        predictor=predictor,
        extractor=extractor,
        rag=RAGPipeline(embedder, store),
        ocr_provider=ocr_provider,
    )
