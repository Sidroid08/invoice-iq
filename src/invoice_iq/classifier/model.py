"""The PyTorch model: a fastText-style EmbeddingBag text classifier.

A learned token embedding is averaged across the document (EmbeddingBag, mean)
and passed through a linear layer to class logits. Small, fast on CPU, and a
genuinely trained model — not a stub or a lookup table.
"""

from __future__ import annotations

import torch
from torch import nn


class DocumentClassifier(nn.Module):
    """EmbeddingBag + linear classifier over document tokens."""

    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text, offsets)
        logits: torch.Tensor = self.fc(embedded)
        return logits
