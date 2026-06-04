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
| 2 | Ingestion + extraction (local) | done |
| 3 | PyTorch classifier (trained) | done |
| 4 | RAG pipeline (embed + Chroma + Q&A) | done |
| 5 | Agentic LangGraph workflow | done |
| 6 | FastAPI serving + monitoring + Docker/CI | done |
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

## Ingestion + extraction (Phase 2)

PDFs become validated `Invoice` JSON through two provider-backed stages, each
behind a `Protocol` so a GCP swap-in is config, not a rewrite:

```
PDF ──► OCRProvider ──► OCRResult ──► Extractor ──► Invoice (validated)
        local: pdfplumber            rule_based (default, deterministic)
        gcp:   Document AI (P7)      llm: Claude API (opt-in)
```

- **`LocalOCRProvider`** ([ingestion/local_ocr.py](src/invoice_iq/ingestion/local_ocr.py)) —
  pdfplumber text extraction; optional Tesseract fallback for scanned PDFs
  (`[ocr]` extra, off by default — needs the Tesseract binary).
- **`RuleBasedExtractor`** ([extraction/rule_based.py](src/invoice_iq/extraction/rule_based.py)) —
  deterministic, dependency-free, **always-on default**; tests/CI never need a key.
- **`LLMExtractor`** ([extraction/llm_extractor.py](src/invoice_iq/extraction/llm_extractor.py)) —
  opt-in Claude-API path (structured tool-use) for messy layouts; enabled only
  with `ENABLE_LLM_EXTRACTION=true` + `ANTHROPIC_API_KEY`.
- **Synthetic data** ([synthetic.py](src/invoice_iq/synthetic.py)) — generates
  realistic invoice/receipt/contract PDFs (the contrast classes for the Phase 3
  classifier) and is the single source of truth for the on-page layout, so the
  generator and extractor can't drift. Correctness is proven by a
  **`Invoice → PDF → OCR → Invoice` round-trip test** over random invoices.

```bash
# Regenerate the labelled training corpus (gitignored output)
python scripts/generate_synthetic_data.py --n 30 --out data/generated --seed 42
```

Tiny committed samples live in [`data/samples/`](data/samples) for the demo.

---

## Document classifier (Phase 3)

A **genuinely trained PyTorch** model classifies document text into
`invoice` / `receipt` / `contract`. It is a fastText-style classifier — a learned
`nn.EmbeddingBag` (mean-pooled token embeddings) → linear layer — trained with
Adam + cross-entropy ([model.py](src/invoice_iq/classifier/model.py),
[train.py](src/invoice_iq/classifier/train.py)). Metrics (accuracy, macro-F1,
confusion matrix) are computed in pure numpy and written to `models/metrics.json`,
which is auto-synced into the section below.

```bash
python -m invoice_iq.classifier.train --epochs 20 --out models   # train + write metrics.json
python scripts/sync_metrics_readme.py                            # refresh README metrics
```

The model is evaluated on a **held-out test set generated with a different RNG
seed** than training (no leakage). End-to-end at inference: `PDF → OCR → classifier
→ DocumentType`. The test-suite trains a small model on every run and asserts it
**beats the 1/3 random baseline by a wide margin** — no stubbing.

> **Scope honesty:** training/eval data is *synthetic* (the Phase 2 generator), and
> the three classes have distinct vocabularies, so the model separates them cleanly
> (hence the perfect score below). The value demonstrated is the **end-to-end MLOps
> path** — train → checkpoint → metrics → serve — not a hard NLP benchmark. The same
> pipeline accepts real labelled PDFs by swapping the data source.

---

## Classifier metrics

<!-- METRICS:START -->
**Accuracy:** 100.0% &nbsp;|&nbsp; **Macro-F1:** 1.000 &nbsp;|&nbsp; **Test samples:** 120

| actual \ pred | invoice | receipt | contract |
|---|---|---|---|
| **invoice** | 40 | 0 | 0 |
| **receipt** | 0 | 40 | 0 |
| **contract** | 0 | 0 | 40 |

_Last trained: 2026-06-04T18:05:38+00:00_
<!-- METRICS:END -->

> This block is auto-generated from `models/metrics.json` by
> `scripts/sync_metrics_readme.py` (run in CI), so the numbers shown are the real,
> last-trained results — never hand-edited.

---

## RAG pipeline (Phase 4)

Extracted invoices are indexed and made queryable, each stage behind a `Protocol`
for the Phase 7 Vertex swap:

```
Invoice ─► chunk_invoice ─► EmbeddingProvider ─► VectorStore ─► ask() ─► Answer + citations
           (summary /        minilm (semantic)    Chroma          retrieve top-k,
            vendor /          hashing (tests)      (cosine)        extractive answer
            line-item)
```

- **Field-aware chunking** ([chunking.py](src/invoice_iq/rag/chunking.py)) — each
  invoice becomes self-contained `summary` / `vendor` / `line_item` chunks carrying
  citation metadata (`doc_id`, `vendor`, `invoice_number`), which retrieve far
  better than naive text splitting.
- **`EmbeddingProvider`** ([embeddings.py](src/invoice_iq/rag/embeddings.py)) —
  `SentenceTransformerEmbedder` (pinned **all-MiniLM-L6-v2**, real semantic vectors)
  and a deterministic dependency-free `HashingEmbedder` for fast, hermetic tests.
- **`VectorStore`** ([vector_store.py](src/invoice_iq/rag/vector_store.py)) —
  Chroma (cosine space), in-memory by default or persistent on disk.
- **`RAGPipeline`** ([qa.py](src/invoice_iq/rag/qa.py)) — index invoices, then
  `ask("How much do we owe Acme Corp?")` retrieves the right document and returns an
  **extractive answer with cited sources**. The Phase 5 agent adds LLM synthesis on
  top of these same retrieved chunks.

The **headline test** proves real MiniLM retrieval pulls the correct invoice and the
answer contains the expected total. The pinned model (~80 MB) downloads once on
first use and is cached (CI caches `~/.cache/huggingface`).

> **Local note (this machine only):** model downloads go through a TLS-intercepting
> proxy, so `.\tasks.ps1` points `SSL_CERT_FILE` at the exported Windows CA bundle.
> On a normal machine / CI this is a no-op.

---

## Agentic workflow (Phase 5)

The LangGraph agent composes the local pipeline into one typed, testable workflow:

```
PDF -> ingest/OCR -> classify -> extract invoice -> optional RAG answer -> recommend
                         |              |
                         |              +-> review on extraction/validation errors
                         +-> route non-invoices to their own workflow
```

- **Typed state** ([state.py](src/invoice_iq/agent/state.py)) - `AgentState`
  captures graph inputs, intermediate artifacts, trace events, and accumulated
  errors; `AgentResult` is the final Pydantic response contract.
- **Tool layer** ([tools.py](src/invoice_iq/agent/tools.py)) - deterministic
  classify/extract/answer/recommend functions plus `StructuredTool` wrappers for
  a future LLM-driven variant.
- **Graph orchestration** ([graph.py](src/invoice_iq/agent/graph.py)) - conditional
  edges skip extraction for non-invoices, skip answering when no question is asked,
  and route failed or high-value invoices to human review.

The Phase 5 tests run the real graph end-to-end on generated PDFs, including the
invoice + question path, no-question path, non-invoice routing, high-value review,
and extraction-failure review.

---

## API serving (Phase 6)

The local pipeline is served through FastAPI with dependency injection for every
provider, so tests can use hermetic components while production loads the trained
checkpoint from `models/classifier.pt`.

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Service status, provider wiring, vector count |
| `POST /ingest` | Upload PDF -> `RawDocument` + `OCRResult` |
| `POST /classify` | Upload PDF -> document type + confidence |
| `POST /extract` | Upload invoice PDF -> validated `Invoice` |
| `POST /ask` | Upload invoice PDF + question -> RAG answer with citations |
| `POST /agent` | Upload PDF + optional question -> full LangGraph `AgentResult` |
| `GET /metrics` | Latency, error-rate, endpoint, and prediction counters |

```bash
# Local API (requires models/classifier.pt; run the Phase 3 train command if absent)
make serve

# Container API (build trains a local classifier inside the image)
make docker-up
```

Windows equivalents: `.\tasks.ps1 serve`, `.\tasks.ps1 docker-build`, and
`.\tasks.ps1 docker-up`.

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
#    Install the CPU-only PyTorch first to avoid the large CUDA wheel:
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
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
| Run API | `make serve` | `.\tasks.ps1 serve` |
| Docker build | `make docker-build` | `.\tasks.ps1 docker-build` |
| Docker compose | `make docker-up` | `.\tasks.ps1 docker-up` |
| Sync metrics → README | `make sync-metrics` | `.\tasks.ps1 sync-metrics` |

Both runners execute identical commands; Windows has no `make`, so `tasks.ps1` mirrors it.

---

## Project layout

See [PLAN.md](PLAN.md) for the full structure and phased plan. Key directories:

```
src/invoice_iq/   # the package (schemas, ingestion, extraction, classifier, rag, agent, serving)
tests/            # pytest suite (one module per component)
scripts/          # synthetic-data generator, metrics sync, demo
Dockerfile        # non-root API image; trains a local classifier during build
docker-compose.yml # API service + persistent Chroma/Hugging Face cache volumes
.github/workflows # CI: ruff + mypy + pytest on Python 3.11
```

---

## License

MIT — see [LICENSE](LICENSE).
