"""Optional LLM-backed extraction via the Claude API (structured tool output).

This is the *opt-in* path for messy, free-form invoices that the deterministic
rule-based extractor can't handle. It is disabled by default: it requires both
`ENABLE_LLM_EXTRACTION=true` and an `ANTHROPIC_API_KEY`. The `anthropic` package
is imported lazily (declared under the `[llm]` extra), so the base install and
the test-suite never need it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from invoice_iq.extraction.base import ExtractionError
from invoice_iq.schemas.documents import OCRResult
from invoice_iq.schemas.invoice import Invoice

if TYPE_CHECKING:
    from config.settings import Settings

DEFAULT_MODEL = "claude-3-5-sonnet-latest"

# Tool schema the model must fill — mirrors the `Invoice` contract (amounts as
# strings to preserve Decimal precision; dates as ISO-8601).
_INVOICE_TOOL: dict[str, Any] = {
    "name": "emit_invoice",
    "description": "Return the structured invoice extracted from the document text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "invoice_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
            "due_date": {"type": ["string", "null"]},
            "currency": {"type": "string", "description": "3-letter ISO 4217 code"},
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": ["string", "null"]},
                    "tax_id": {"type": ["string", "null"]},
                    "email": {"type": ["string", "null"]},
                },
                "required": ["name"],
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "string"},
                        "unit_price": {"type": "string"},
                        "line_total": {"type": "string"},
                    },
                    "required": ["description", "quantity", "unit_price", "line_total"],
                },
            },
            "subtotal": {"type": "string"},
            "tax": {"type": ["string", "null"]},
            "total": {"type": "string"},
        },
        "required": [
            "invoice_number",
            "invoice_date",
            "currency",
            "line_items",
            "subtotal",
            "total",
        ],
    },
}


class LLMExtractor:
    """Extracts invoices from arbitrary text using the Claude API."""

    name = "llm"

    def __init__(self, api_key: str, *, model: str = DEFAULT_MODEL, max_tokens: int = 2000) -> None:
        if not api_key:
            raise RuntimeError("LLMExtractor requires a non-empty ANTHROPIC_API_KEY")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMExtractor:
        if not settings.llm_extraction_ready:
            raise RuntimeError(
                "LLM extraction is not enabled. Set ENABLE_LLM_EXTRACTION=true and "
                "provide ANTHROPIC_API_KEY in the environment / .env."
            )
        assert settings.anthropic_api_key is not None  # guaranteed by llm_extraction_ready
        return cls(api_key=settings.anthropic_api_key)

    def _get_client(self) -> Any:  # noqa: ANN401 — anthropic client is untyped (Any)
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415 — optional dependency, imported lazily
            except ImportError as exc:  # pragma: no cover - exercised only without extra
                raise RuntimeError(
                    "LLM extraction requires the '[llm]' extra: pip install -e '.[llm]'"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def extract(self, ocr: OCRResult) -> Invoice:
        client = self._get_client()
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            tools=[_INVOICE_TOOL],
            tool_choice={"type": "tool", "name": _INVOICE_TOOL["name"]},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract the invoice into the emit_invoice tool. Use exact "
                        "values from the document. Document text:\n\n" + ocr.full_text
                    ),
                }
            ],
        )
        payload = self._extract_tool_payload(message)
        return self._to_invoice(payload)

    @staticmethod
    def _extract_tool_payload(message: Any) -> dict[str, Any]:  # noqa: ANN401 — untyped SDK response
        for block in getattr(message, "content", []):
            if getattr(block, "type", None) == "tool_use":
                data: dict[str, Any] = block.input
                return data
        raise ExtractionError("LLM response contained no tool_use block")

    @staticmethod
    def _to_invoice(payload: dict[str, Any]) -> Invoice:
        currency = str(payload.get("currency", "")).upper()

        def money(value: object) -> dict[str, Any]:
            return {"amount": str(value), "currency": currency}

        items = [
            {
                "description": it["description"],
                "quantity": str(it["quantity"]),
                "unit_price": money(it["unit_price"]),
                "line_total": money(it["line_total"]),
            }
            for it in payload.get("line_items", [])
        ]
        doc = {
            "invoice_number": payload["invoice_number"],
            "vendor": payload["vendor"],
            "invoice_date": payload["invoice_date"],
            "due_date": payload.get("due_date"),
            "currency": currency,
            "line_items": items,
            "subtotal": money(payload["subtotal"]),
            "tax": money(payload["tax"]) if payload.get("tax") else None,
            "total": money(payload["total"]),
        }
        try:
            return Invoice.model_validate(doc)
        except ValueError as exc:
            raise ExtractionError(f"LLM output failed validation: {exc}") from exc
