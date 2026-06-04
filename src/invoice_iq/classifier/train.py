"""Train the document classifier, checkpoint it, and emit metrics.json.

Train and test sets are generated with *different* RNG seeds so the model is
always evaluated on unseen documents (no leakage). Reproducible via a single
seed. Run as a module:

    python -m invoice_iq.classifier.train --epochs 20 --out models
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from invoice_iq.classifier.dataset import (
    LABELS,
    DocumentDataset,
    Vocabulary,
    collate_batch,
    generate_labeled_texts,
)
from invoice_iq.classifier.evaluate import ClassificationMetrics, compute_metrics
from invoice_iq.classifier.model import DocumentClassifier

CHECKPOINT_NAME = "classifier.pt"
METRICS_NAME = "metrics.json"


@dataclass
class TrainConfig:
    n_train_per_class: int = 80
    n_test_per_class: int = 40
    epochs: int = 20
    embed_dim: int = 64
    lr: float = 0.01
    batch_size: int = 16
    seed: int = 42


def _make_loader(
    dataset: DocumentDataset, batch_size: int, *, shuffle: bool, seed: int
) -> DataLoader[tuple[list[int], int]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        generator=generator,
    )


@torch.no_grad()
def _predict_dataset(
    model: DocumentClassifier, loader: DataLoader[tuple[list[int], int]]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    model.eval()
    trues: list[int] = []
    preds: list[int] = []
    for text, offsets, labels in loader:
        logits = model(text, offsets)
        batch_preds = torch.argmax(logits, dim=1)
        trues.extend(int(x) for x in labels)
        preds.extend(int(x) for x in batch_preds)
    return np.array(trues, dtype=np.int64), np.array(preds, dtype=np.int64)


def train_classifier(
    config: TrainConfig | None = None,
) -> tuple[DocumentClassifier, Vocabulary, ClassificationMetrics]:
    """Train on synthetic data and return (model, vocab, test metrics)."""
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)

    train_samples = generate_labeled_texts(cfg.n_train_per_class, seed=cfg.seed)
    test_samples = generate_labeled_texts(cfg.n_test_per_class, seed=cfg.seed + 1000)

    vocab = Vocabulary.build(text for text, _ in train_samples)
    train_ds = DocumentDataset(train_samples, vocab)
    test_ds = DocumentDataset(test_samples, vocab)

    train_loader = _make_loader(train_ds, cfg.batch_size, shuffle=True, seed=cfg.seed)
    test_loader = _make_loader(test_ds, cfg.batch_size, shuffle=False, seed=cfg.seed)

    model = DocumentClassifier(len(vocab), num_classes=len(LABELS), embed_dim=cfg.embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(cfg.epochs):
        model.train()
        for text, offsets, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(text, offsets), labels)
            loss.backward()
            optimizer.step()

    y_true, y_pred = _predict_dataset(model, test_loader)
    metrics = compute_metrics(y_true, y_pred, num_classes=len(LABELS))
    return model, vocab, metrics


def save_checkpoint(model: DocumentClassifier, vocab: Vocabulary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "itos": vocab.itos,
            "vocab_size": model.vocab_size,
            "num_classes": model.num_classes,
            "embed_dim": model.embed_dim,
            "labels": [label.value for label in LABELS],
        },
        path,
    )


def write_metrics(metrics: ClassificationMetrics, cfg: TrainConfig, path: Path) -> None:
    payload = {
        **metrics,
        "labels": [label.value for label in LABELS],
        "trained_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "config": asdict(cfg),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the document classifier")
    parser.add_argument("--n-train", type=int, default=TrainConfig.n_train_per_class)
    parser.add_argument("--n-test", type=int, default=TrainConfig.n_test_per_class)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--embed-dim", type=int, default=TrainConfig.embed_dim)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        n_train_per_class=args.n_train,
        n_test_per_class=args.n_test,
        epochs=args.epochs,
        embed_dim=args.embed_dim,
        lr=args.lr,
        seed=args.seed,
    )
    model, vocab, metrics = train_classifier(cfg)

    out: Path = args.out
    save_checkpoint(model, vocab, out / CHECKPOINT_NAME)
    write_metrics(metrics, cfg, out / METRICS_NAME)

    print(
        f"[train] accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"n_test={metrics['n_test']}"
    )
    print(f"[train] checkpoint -> {out / CHECKPOINT_NAME}")
    print(f"[train] metrics    -> {out / METRICS_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
