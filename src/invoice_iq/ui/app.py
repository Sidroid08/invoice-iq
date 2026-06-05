"""Streamlit demo UI for invoice-iq.

Upload an invoice PDF and watch the full pipeline run through the live API:
classify → extract → ask → recommend. Driven entirely by the FastAPI `/agent`
endpoint via `AgentAPIClient`.

Run it::

    pip install -e ".[demo]"
    uvicorn invoice_iq.serving.app:create_app --factory      # in one terminal
    streamlit run src/invoice_iq/ui/app.py                    # in another

Set INVOICE_IQ_API_URL to point at a non-default API location.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import streamlit as st

from invoice_iq.schemas.api import AgentRunResponse
from invoice_iq.ui.client import DEFAULT_BASE_URL, AgentAPIClient
from invoice_iq.ui.view import invoice_line_item_rows, recommendation_badge

API_URL = os.environ.get("INVOICE_IQ_API_URL", DEFAULT_BASE_URL)
SAMPLE_PDF = Path("data/samples/sample_invoice.pdf")


def _sidebar() -> None:
    st.sidebar.header("Service")
    st.sidebar.caption(f"API: `{API_URL}`")
    if st.sidebar.button("Check health"):
        try:
            health = AgentAPIClient(API_URL).health()
            st.sidebar.success(f"ok · v{health.get('version', '?')}")
            st.sidebar.json(health.get("providers", {}))
        except httpx.HTTPError as exc:
            st.sidebar.error(f"API unreachable: {exc}")


def _render_classification(result: AgentRunResponse) -> None:
    st.subheader("1 · Classification")
    col1, col2 = st.columns(2)
    col1.metric("Document type", result.document_type.value)
    col2.metric("Confidence", f"{result.classification_confidence:.1%}")


def _render_extraction(result: AgentRunResponse) -> None:
    st.subheader("2 · Extraction")
    invoice = result.invoice
    if invoice is None:
        st.info("No invoice extracted — the document is not an invoice.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Vendor", invoice.vendor.name)
    col2.metric("Invoice #", invoice.invoice_number)
    col3.metric("Total", str(invoice.total))
    st.table(invoice_line_item_rows(invoice))
    st.caption(
        f"Subtotal {invoice.subtotal} · Tax {invoice.tax or '—'} · "
        f"Date {invoice.invoice_date.isoformat()}"
    )


def _render_answer(result: AgentRunResponse) -> None:
    if result.answer is None:
        return
    st.subheader("3 · Question & Answer (RAG)")
    st.write(result.answer.answer)
    with st.expander("Retrieved sources"):
        for source in result.answer.sources:
            st.markdown(f"- *(score {source.score:.2f})* {source.text}")


def _render_recommendation(result: AgentRunResponse) -> None:
    st.subheader("4 · Recommended action")
    label, style = recommendation_badge(result.recommendation)
    getattr(st, style)(f"{label} — {result.rationale}")
    with st.expander("Agent trace"):
        for step in result.trace:
            st.markdown(f"- {step}")


def _run(pdf_bytes: bytes, filename: str, question: str) -> None:
    try:
        result = AgentAPIClient(API_URL).run_agent(pdf_bytes, filename, question or None)
    except httpx.HTTPError as exc:
        st.error(f"Request failed: {exc}")
        return
    _render_classification(result)
    _render_extraction(result)
    _render_answer(result)
    _render_recommendation(result)


def main() -> None:
    st.set_page_config(page_title="invoice-iq demo", page_icon="🧾", layout="centered")
    st.title("🧾 invoice-iq")
    st.caption("Upload an invoice PDF → classify → extract → ask → recommend.")
    _sidebar()

    uploaded = st.file_uploader("Invoice PDF", type=["pdf"])
    question = st.text_input("Optional question", value="What is the total?")

    use_sample = False
    if uploaded is None and SAMPLE_PDF.exists():
        use_sample = st.checkbox(f"Use bundled sample ({SAMPLE_PDF.name})")

    if st.button("Run pipeline", type="primary"):
        if uploaded is not None:
            _run(uploaded.getvalue(), uploaded.name, question)
        elif use_sample:
            _run(SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name, question)
        else:
            st.warning("Upload a PDF or tick the sample checkbox first.")


main()
