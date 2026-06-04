"""Load a trained checkpoint and classify document text → (DocumentType, confidence)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from invoice_iq.classifier.dataset import Vocabulary
from invoice_iq.classifier.model import DocumentClassifier
from invoice_iq.schemas.documents import DocumentType, OCRResult


@runtime_checkable
class DocumentTypePredictor(Protocol):
    """Predicts document type from text or OCR output."""

    def predict(self, text: str) -> tuple[DocumentType, float]: ...

    def predict_ocr(self, ocr: OCRResult) -> tuple[DocumentType, float]: ...


@dataclass
class Predictor:
    """Wraps a trained model + vocabulary for inference."""

    model: DocumentClassifier
    vocab: Vocabulary
    labels: list[DocumentType]

    def predict(self, text: str) -> tuple[DocumentType, float]:
        """Return the predicted class and its softmax confidence in [0, 1]."""
        if not text.strip():
            return DocumentType.UNKNOWN, 0.0
        ids = self.vocab.encode(text)
        text_tensor = torch.tensor(ids, dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(text_tensor, offsets)
            probs = torch.softmax(logits, dim=1)
            confidence, index = torch.max(probs, dim=1)
        return self.labels[int(index.item())], float(confidence.item())

    def predict_ocr(self, ocr: OCRResult) -> tuple[DocumentType, float]:
        return self.predict(ocr.full_text)


def load_classifier(path: Path | str) -> Predictor:
    """Reconstruct a `Predictor` from a checkpoint saved by `train.save_checkpoint`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no classifier checkpoint at {path}")
    # weights_only=False: this is our own trusted artifact (stores vocab + labels).
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    vocab = Vocabulary(list(checkpoint["itos"]))
    model = DocumentClassifier(
        vocab_size=int(checkpoint["vocab_size"]),
        num_classes=int(checkpoint["num_classes"]),
        embed_dim=int(checkpoint["embed_dim"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    labels = [DocumentType(value) for value in checkpoint["labels"]]
    return Predictor(model=model, vocab=vocab, labels=labels)
