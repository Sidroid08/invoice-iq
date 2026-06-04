"""Extraction interfaces shared by rule-based and LLM extractors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from invoice_iq.schemas.documents import OCRResult
from invoice_iq.schemas.invoice import Invoice


class ExtractionError(ValueError):
    """Raised when text cannot be parsed into a valid `Invoice`."""


@runtime_checkable
class Extractor(Protocol):
    """Maps an `OCRResult` to a validated `Invoice`.

    Implementations: `RuleBasedExtractor` (deterministic, always-on default) and
    `LLMExtractor` (optional, Claude-API-backed). Both raise `ExtractionError`
    when the document cannot be parsed into a consistent invoice.
    """

    @property
    def name(self) -> str:
        """Short extractor id (e.g. 'rule_based', 'llm')."""
        ...

    def extract(self, ocr: OCRResult) -> Invoice:
        """Parse `ocr` into an `Invoice` or raise `ExtractionError`."""
        ...
