"""Tests for the PyTorch document classifier (Phase 3).

These genuinely train a (small) model and assert it *learns* — accuracy must beat
the 1/3 random baseline by a wide margin. No stubbing.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from invoice_iq.classifier import LABELS, Predictor, load_classifier
from invoice_iq.classifier.evaluate import compute_metrics, confusion_matrix
from invoice_iq.classifier.train import TrainConfig, save_checkpoint, train_classifier
from invoice_iq.schemas.documents import DocumentType
from invoice_iq.synthetic import (
    invoice_to_lines,
    random_contract_lines,
    random_invoice,
    random_receipt_lines,
)

# Small, fast config shared across tests (trains in ~1s on CPU).
_CFG = TrainConfig(n_train_per_class=20, n_test_per_class=10, epochs=10, seed=7)


@pytest.fixture(scope="module")
def trained() -> Predictor:
    model, vocab, _ = train_classifier(_CFG)
    return Predictor(model=model, vocab=vocab, labels=list(LABELS))


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #
def test_model_actually_learns() -> None:
    _, _, metrics = train_classifier(_CFG)
    # Random baseline for 3 balanced classes is ~0.33; require a wide margin.
    assert metrics["accuracy"] > 0.8
    assert metrics["macro_f1"] > 0.8
    assert metrics["n_test"] == 3 * _CFG.n_test_per_class


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def test_predicts_each_class(trained: Predictor) -> None:
    rng = random.Random(123)
    invoice_text = "\n".join(invoice_to_lines(random_invoice(rng)))
    receipt_text = "\n".join(random_receipt_lines(rng))
    contract_text = "\n".join(random_contract_lines(rng))

    assert trained.predict(invoice_text)[0] is DocumentType.INVOICE
    assert trained.predict(receipt_text)[0] is DocumentType.RECEIPT
    assert trained.predict(contract_text)[0] is DocumentType.CONTRACT


def test_prediction_confidence_in_range(trained: Predictor) -> None:
    label, confidence = trained.predict("RECEIPT Store: QuickMart TOTAL $5.00 Payment: VISA")
    assert isinstance(label, DocumentType)
    assert 0.0 <= confidence <= 1.0


def test_empty_text_is_unknown(trained: Predictor) -> None:
    label, confidence = trained.predict("   ")
    assert label is DocumentType.UNKNOWN
    assert confidence == 0.0


# --------------------------------------------------------------------------- #
# Checkpoint round-trip
# --------------------------------------------------------------------------- #
def test_checkpoint_round_trip(tmp_path: Path, trained: Predictor) -> None:
    ckpt = tmp_path / "classifier.pt"
    save_checkpoint(trained.model, trained.vocab, ckpt)
    reloaded = load_classifier(ckpt)

    text = "INVOICE Invoice Number: INV-1 Vendor: Acme Subtotal: 10.00 Total: 10.00"
    assert reloaded.predict(text) == trained.predict(text)


def test_load_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_classifier(tmp_path / "nope.pt")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_compute_metrics_known_values() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1, 2, 2], dtype=np.int64)
    metrics = compute_metrics(y_true, y_pred, num_classes=3)
    assert metrics["accuracy"] == pytest.approx(0.8333, abs=1e-4)
    assert metrics["macro_f1"] == pytest.approx(0.8222, abs=1e-4)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 2]]
    assert metrics["n_test"] == 6


def test_confusion_matrix_shape() -> None:
    y = np.array([0, 1, 2], dtype=np.int64)
    matrix = confusion_matrix(y, y, num_classes=3)
    assert matrix.shape == (3, 3)
    assert int(matrix.trace()) == 3  # perfect predictions on the diagonal
