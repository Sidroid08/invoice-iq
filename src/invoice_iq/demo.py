"""One-command local demo for invoice-iq."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from invoice_iq.agent import AgentDeps, NextAction, run_agent
from invoice_iq.classifier import LABELS, Predictor, load_classifier
from invoice_iq.classifier.train import CHECKPOINT_NAME, TrainConfig, train_classifier
from invoice_iq.extraction import RuleBasedExtractor
from invoice_iq.rag import ChromaVectorStore, HashingEmbedder, RAGPipeline
from invoice_iq.schemas.documents import DocumentType

DEFAULT_PDF = Path("data/samples/sample_invoice.pdf")
DEFAULT_QUESTION = "What is the total?"


class DemoSummary(BaseModel):
    """Recruiter-friendly summary of one end-to-end run."""

    model_config = ConfigDict(extra="forbid")

    pdf_path: str
    predictor_source: Literal["checkpoint", "ephemeral-trained"]
    document_type: DocumentType
    classification_confidence: float = Field(ge=0.0, le=1.0)
    invoice_number: str | None = None
    vendor: str | None = None
    total: str | None = None
    question: str | None = None
    answer: str | None = None
    recommendation: NextAction
    trace: list[str]


def _load_or_train_predictor(
    checkpoint: Path,
    *,
    train_if_missing: bool,
) -> tuple[Predictor, Literal["checkpoint", "ephemeral-trained"]]:
    if checkpoint.is_file():
        return load_classifier(checkpoint), "checkpoint"
    if not train_if_missing:
        raise FileNotFoundError(
            f"no classifier checkpoint at {checkpoint}; run "
            "`python -m invoice_iq.classifier.train --epochs 20 --out models`"
        )
    model, vocab, _ = train_classifier(
        TrainConfig(n_train_per_class=20, n_test_per_class=5, epochs=10, seed=7)
    )
    return Predictor(model=model, vocab=vocab, labels=list(LABELS)), "ephemeral-trained"


def run_demo(
    pdf_path: Path | str = DEFAULT_PDF,
    *,
    question: str | None = DEFAULT_QUESTION,
    checkpoint: Path | str = Path("models") / CHECKPOINT_NAME,
    train_if_missing: bool = True,
) -> DemoSummary:
    """Run the full local agent on a PDF and return a compact summary."""
    pdf = Path(pdf_path)
    predictor, predictor_source = _load_or_train_predictor(
        Path(checkpoint), train_if_missing=train_if_missing
    )
    deps = AgentDeps(
        predictor=predictor,
        extractor=RuleBasedExtractor(),
        rag=RAGPipeline(
            HashingEmbedder(),
            ChromaVectorStore(collection_name=f"demo_{uuid4().hex[:8]}"),
        ),
    )
    result = run_agent(deps, pdf, question=question)
    invoice = result.invoice
    return DemoSummary(
        pdf_path=str(pdf),
        predictor_source=predictor_source,
        document_type=result.document_type,
        classification_confidence=result.classification_confidence,
        invoice_number=invoice.invoice_number if invoice else None,
        vendor=invoice.vendor.name if invoice else None,
        total=str(invoice.total) if invoice else None,
        question=question,
        answer=result.answer.answer if result.answer else None,
        recommendation=result.recommendation,
        trace=result.trace,
    )


def render_human(summary: DemoSummary) -> str:
    """Render a terminal-friendly summary."""
    lines = [
        "invoice-iq demo",
        "===============",
        f"PDF: {summary.pdf_path}",
        f"Classifier: {summary.predictor_source}",
        (
            f"Classification: {summary.document_type.value} "
            f"({summary.classification_confidence:.2f})"
        ),
    ]
    if summary.invoice_number:
        lines.extend(
            [
                f"Invoice: {summary.invoice_number}",
                f"Vendor: {summary.vendor}",
                f"Total: {summary.total}",
            ]
        )
    if summary.question:
        lines.append(f"Question: {summary.question}")
    if summary.answer:
        lines.append(f"Answer: {summary.answer}")
    lines.extend(
        [
            f"Recommendation: {summary.recommendation.value}",
            "Trace:",
            *[f"  - {step}" for step in summary.trace],
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local invoice-iq demo.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF to process")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question for RAG")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models") / CHECKPOINT_NAME,
        help="classifier checkpoint path",
    )
    parser.add_argument(
        "--no-train-if-missing",
        action="store_true",
        help="fail instead of training a tiny in-memory classifier if the checkpoint is absent",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)

    summary = run_demo(
        args.pdf,
        question=args.question,
        checkpoint=args.checkpoint,
        train_if_missing=not args.no_train_if_missing,
    )
    if args.json:
        print(summary.model_dump_json(indent=2))
    else:
        print(render_human(summary))
    return 0
