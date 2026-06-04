"""Dataset, tokenizer, and vocabulary for the document classifier.

Training text is produced directly from the synthetic generators (same text an
OCR pass yields from the rendered PDFs), so training is fast and reproducible
while still matching what the model sees at inference on real `OCRResult` text.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable

import torch
from torch.utils.data import Dataset

from invoice_iq.schemas.documents import DocumentType
from invoice_iq.synthetic import (
    invoice_to_lines,
    random_contract_lines,
    random_invoice,
    random_receipt_lines,
)

# The three classes the model distinguishes (UNKNOWN is reserved for low-confidence
# predictions at inference time, never a training label).
LABELS: tuple[DocumentType, ...] = (
    DocumentType.INVOICE,
    DocumentType.RECEIPT,
    DocumentType.CONTRACT,
)
LABEL_TO_INDEX: dict[DocumentType, int] = {label: i for i, label in enumerate(LABELS)}

_TOKEN_RE = re.compile(r"[a-z0-9@.]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer used everywhere (train + inference)."""
    return _TOKEN_RE.findall(text.lower())


def generate_labeled_texts(n_per_class: int, seed: int) -> list[tuple[str, int]]:
    """Generate ``n_per_class`` synthetic documents per class as (text, label_idx)."""
    rng = random.Random(seed)
    samples: list[tuple[str, int]] = []
    for _ in range(n_per_class):
        invoice_text = "\n".join(invoice_to_lines(random_invoice(rng)))
        receipt_text = "\n".join(random_receipt_lines(rng))
        contract_text = "\n".join(random_contract_lines(rng))
        samples.append((invoice_text, LABEL_TO_INDEX[DocumentType.INVOICE]))
        samples.append((receipt_text, LABEL_TO_INDEX[DocumentType.RECEIPT]))
        samples.append((contract_text, LABEL_TO_INDEX[DocumentType.CONTRACT]))
    rng.shuffle(samples)
    return samples


class Vocabulary:
    """Maps tokens to integer ids. Index 0 is the reserved ``<unk>`` token."""

    UNK = "<unk>"

    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> Vocabulary:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        vocab = [tok for tok, count in sorted(counter.items()) if count >= min_freq]
        return cls([cls.UNK, *vocab])

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi.get(tok, 0) for tok in tokenize(text)]
        return ids or [0]  # never emit an empty bag

    def __len__(self) -> int:
        return len(self.itos)


class DocumentDataset(Dataset[tuple[list[int], int]]):
    """Holds encoded (token-id list, label) pairs."""

    def __init__(self, samples: list[tuple[str, int]], vocab: Vocabulary) -> None:
        self.vocab = vocab
        self.data: list[tuple[list[int], int]] = [
            (vocab.encode(text), label) for text, label in samples
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.data[index]


def collate_batch(
    batch: list[tuple[list[int], int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length token lists into EmbeddingBag (text, offsets, labels)."""
    labels: list[int] = []
    flat_tokens: list[int] = []
    lengths: list[int] = [0]
    for ids, label in batch:
        labels.append(label)
        flat_tokens.extend(ids)
        lengths.append(len(ids))
    text = torch.tensor(flat_tokens, dtype=torch.long)
    offsets = torch.tensor(lengths[:-1], dtype=torch.long).cumsum(dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return text, offsets, label_tensor
