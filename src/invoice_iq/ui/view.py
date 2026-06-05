"""Pure presentation helpers for the Streamlit demo (no Streamlit import).

Kept separate from `app.py` so they are unit-testable without a Streamlit runtime.
"""

from __future__ import annotations

from invoice_iq.agent.state import NextAction
from invoice_iq.schemas.invoice import Invoice

# Maps a recommendation to a (display label, Streamlit status style).
# style is one of: "success", "warning", "info", "error".
_BADGES: dict[NextAction, tuple[str, str]] = {
    NextAction.APPROVE: ("✅ Approve for payment", "success"),
    NextAction.REVIEW: ("⚠️ Flag for human review", "warning"),
    NextAction.REQUEST_INFO: ("ℹ️ Request more information", "info"),
    NextAction.ROUTE_NON_INVOICE: ("↪️ Route to the correct workflow", "info"),
}


def recommendation_badge(action: NextAction) -> tuple[str, str]:
    """Return (label, status_style) for a recommended next action."""
    return _BADGES.get(action, (action.value, "info"))


def invoice_line_item_rows(invoice: Invoice) -> list[dict[str, str]]:
    """Flatten an invoice's line items into table rows for display."""
    return [
        {
            "Description": item.description,
            "Quantity": str(item.quantity),
            "Unit price": str(item.unit_price),
            "Line total": str(item.line_total),
        }
        for item in invoice.line_items
    ]
