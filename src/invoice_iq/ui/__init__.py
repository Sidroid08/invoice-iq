"""Streamlit demo UI for invoice-iq (Phase 8).

The UI is a thin client over the existing FastAPI `/agent` and `/health`
endpoints, so the demo exercises the real serving + agent code path. Pure logic
(the API client and view helpers) lives here and is unit-tested; `app.py` holds
the Streamlit rendering and is launched with `streamlit run`.
"""

from invoice_iq.ui.client import AgentAPIClient
from invoice_iq.ui.view import invoice_line_item_rows, recommendation_badge

__all__ = ["AgentAPIClient", "invoice_line_item_rows", "recommendation_badge"]
