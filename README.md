# invoice-iq — Invoice Intelligence Platform

[![CI](https://github.com/Sidroid08/invoice-iq/actions/workflows/ci.yml/badge.svg)](https://github.com/Sidroid08/invoice-iq/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

> A local-first, MLOps-complete document-AI system for **invoices**: PDF ingestion +
> structured extraction, a **genuinely trained PyTorch** document classifier, a **RAG**
> Q&A pipeline, an **agentic LangGraph** workflow, and **FastAPI** serving — with
> **GCP Vertex AI** swap-ins (incl. deploying the classifier to a Vertex endpoint).

This project is built incrementally, phase by phase. Everything runs **end-to-end on a
laptop with zero cloud spend**; GCP is added behind interfaces only after the local
system is green. See [PLAN.md](PLAN.md) for the full build plan.

---

## Architecture (target)

```
                         ┌──────────────────────────────────────────┐
   invoice.pdf  ───────► │            FastAPI gateway                │
                         │  /ingest /classify /extract /ask /agent   │
                         └───────────────┬──────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
      ┌───────────────┐         ┌─────────────────┐        ┌────────────────┐
      │  Ingestion    │         │  PyTorch         │        │   RAG          │
      │  OCR + layout │────────►│  classifier      │        │  chunk→embed→  │
      │  (local/DocAI)│  text   │  invoice/receipt │        │  Chroma→Q&A    │
      └──────┬────────┘         │  /contract       │        └───────┬────────┘
             │                  │  (local / Vertex)│                │
             ▼                  └─────────────────┘                 │
      ┌───────────────┐                                             │
      │  Extraction   │   validated Pydantic Invoice JSON           │
      │  rules + LLM  │─────────────────────────────────────────────┘
      └───────────────┘
                         ┌──────────────────────────────────────────┐
                         │   LangGraph agent (typed state + tools)   │
                         │  classify → extract → retrieve → answer   │
                         │           → recommend next action         │
                         └──────────────────────────────────────────┘
```

> The diagram describes the target system. Components light up as phases land — see
> **Build status** below.

---

## Build status

| Phase | Component | Status |
|------:|-----------|--------|
| 0 | Scaffold + guardrails (tooling, CI, venv) | done |
| 1 | Pydantic schemas + typed settings | done |
| 2 | Ingestion + extraction (local) | planned |
| 3 | PyTorch classifier (trained) | planned |
| 4 | RAG pipeline | planned |
| 5 | Agentic LangGraph workflow | planned |
| 6 | FastAPI serving + monitoring + Docker/CI | planned |
| 7 | GCP swap-ins (Vertex endpoint first) | planned |
| 8 | Streamlit demo + recruiter packaging | planned |

---

## Data model (the typed spine)

Every stage exchanges **validated Pydantic v2 models** — never raw dicts. Core
contracts (in [`src/invoice_iq/schemas`](src/invoice_iq/schemas)):

| Model | Role | Key guarantees |
|-------|------|----------------|
| `DocumentType` | Class label | `invoice` / `receipt` / `contract` / `unknown` |
| `RawDocument` | Pipeline input | non-blank filename, stable `doc_id`, `extra="forbid"` |
| `OCRResult` | OCR output | non-empty text, per-page list, confidence ∈ [0,1] |
| `Money` | Value object | `Decimal` (never float), 2dp, ISO-4217 currency, **frozen** |
| `LineItem` | Billed line | enforces `quantity × unit_price == line_total` |
| `Vendor` | Issuer | validated email (`EmailStr`) |
| `Invoice` | Extraction output | currency consistency, **subtotal = Σ lines**, **total = subtotal + tax**, `due_date ≥ invoice_date` |

A constructed `Invoice` is guaranteed *internally consistent*, not merely
well-typed — the reconciliation rules live in the model validators. Configuration
is a single typed [`Settings`](config/settings.py) object (pydantic-settings) with
local-first defaults and opt-in GCP/LLM paths.

---

## Classifier metrics

<!-- METRICS:START -->
_Classifier not trained yet. Run the Phase 3 training (`python -m invoice_iq.classifier.train`) to populate this section._
<!-- METRICS:END -->

> This block is auto-generated from `models/metrics.json` by
> `scripts/sync_metrics_readme.py` (run in CI), so the numbers shown are the real,
> last-trained results — never hand-edited.

---

## Setup

Requires **Python 3.11** (pinned via `.python-version`).

```bash
# 1. Create the virtual environment (Windows: py -3.11 -m venv .venv)
python3.11 -m venv .venv

# 2. Activate it
#   Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
#   macOS / Linux:         source .venv/bin/activate

# 3. Install the project + dev tooling
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env      # Windows: copy .env.example .env
```

All configuration is via environment variables; copy `.env.example` to `.env` and edit.
**No secrets are ever committed** — `.env` is gitignored.

---

## Common tasks

| Action | macOS / Linux / CI | Windows |
|--------|--------------------|---------|
| Install | `make install` | `.\tasks.ps1 install` |
| Lint | `make lint` | `.\tasks.ps1 lint` |
| Type-check | `make type` | `.\tasks.ps1 type` |
| Test | `make test` | `.\tasks.ps1 test` |
| Full gate (lint+type+test) | `make check` | `.\tasks.ps1 check` |
| Sync metrics → README | `make sync-metrics` | `.\tasks.ps1 sync-metrics` |

Both runners execute identical commands; Windows has no `make`, so `tasks.ps1` mirrors it.

---

## Project layout

See [PLAN.md](PLAN.md) for the full structure and phased plan. Key directories:

```
src/invoice_iq/   # the package (schemas, ingestion, extraction, classifier, rag, agent, serving)
tests/            # pytest suite (one module per component)
scripts/          # synthetic-data generator, metrics sync, demo
.github/workflows # CI: ruff + mypy + pytest on Python 3.11
```

---

## License

MIT — see [LICENSE](LICENSE).
