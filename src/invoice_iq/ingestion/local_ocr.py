"""Local OCR/parse provider using pdfplumber, with an optional Tesseract fallback.

For text-based PDFs (the common case, and what our synthetic generator emits),
pdfplumber extracts the text layer directly — fast and exact. For *scanned*
PDFs with no text layer, an optional fallback renders each page with PyMuPDF and
runs Tesseract; this path is only active when the `[ocr]` extra is installed and
`enable_ocr_fallback=True`, so the base install needs no system binaries.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from invoice_iq.ingestion.base import PAGE_BREAK, OCRProvider, build_raw_document
from invoice_iq.schemas.documents import OCRResult, RawDocument

# Below this many characters on a page, we treat the text layer as "missing"
# and (if enabled) fall back to image OCR.
_MIN_TEXT_CHARS = 10


class LocalOCRProvider:
    """Extracts text from PDFs locally (no cloud, no cost)."""

    name = "local"

    def __init__(self, *, enable_ocr_fallback: bool = False) -> None:
        self.enable_ocr_fallback = enable_ocr_fallback

    def extract(self, pdf_path: Path, doc_id: str) -> OCRResult:
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {path}")

        pages: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if len(text.strip()) < _MIN_TEXT_CHARS and self.enable_ocr_fallback:
                    text = self._ocr_page_image(page)
                pages.append(text)

        full_text = PAGE_BREAK.join(pages)
        return OCRResult(
            doc_id=doc_id,
            full_text=full_text,
            pages=pages,
            provider=self.name,
        )

    @staticmethod
    def _ocr_page_image(page: object) -> str:
        """Render a page to an image and OCR it (optional `[ocr]` extra)."""
        try:
            import pytesseract  # noqa: PLC0415 — optional dependency, imported lazily
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise RuntimeError(
                "OCR fallback requires the '[ocr]' extra (pymupdf, pytesseract) "
                "and the Tesseract binary on PATH."
            ) from exc
        # pdfplumber pages expose `.to_image()`; render then OCR.
        pil_image: Image.Image = page.to_image(resolution=200).original  # type: ignore[attr-defined]
        text: str = pytesseract.image_to_string(pil_image)
        return text


def ingest_pdf(
    pdf_path: Path,
    provider: OCRProvider | None = None,
) -> tuple[RawDocument, OCRResult]:
    """Build the `RawDocument` and run OCR in one step.

    Returns the metadata record and the OCR output, linked by `doc_id`.
    """
    provider = provider or LocalOCRProvider()
    raw = build_raw_document(pdf_path)
    ocr = provider.extract(Path(pdf_path), raw.doc_id)
    # Backfill page count now that we know it.
    raw = raw.model_copy(update={"num_pages": ocr.page_count})
    return raw, ocr
