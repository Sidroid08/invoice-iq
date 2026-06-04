# PLAN.md — Invoice Intelligence Platform

> A portfolio-grade, MLOps-complete document-AI system for **invoices**, built to
> demonstrate the exact skill set an AI-first company (Quantiphi) hires MLEs for:
> GCP Vertex AI, Document AI–style extraction, RAG, agentic LangGraph orchestration,
> a genuinely fine-tuned PyTorch classifier, FastAPI serving, and full MLOps.

**Working title:** `invoice-iq`
**Single domain:** invoices (with receipt/contract as contrast classes for the classifier).

---

## 1. Guiding principles

1. **Local-first, cloud-second.** Everything runs end-to-end on a laptop with zero
   cloud spend. GCP is a *swap-in* behind interfaces, added only after the local
   system is green. No GCP command runs without your explicit confirmation.
2. **Interfaces over implementations.** Extraction, classification, embeddings, and
   serving each sit behind a small protocol/ABC. Local impl first; Vertex/Document AI
   impl second. Swapping is a config flag, not a rewrite.
3. **Typed and validated everywhere.** Pydantic v2 models are the contract between
   every stage. Nothing passes a raw dict across a boundary.
4. **Genuinely trained, not stubbed.** The PyTorch classifier is trained on real
   (synthetic-but-realistic) data, checkpointed, evaluated, and served. Metrics are
   reported honestly.
5. **Every phase ships green.** Each phase ends with passing `pytest`, an updated
   README, and a single small commit. Tests are written alongside code, not after.
6. **Recruiter-readable.** The README leads with a text architecture diagram, a
   "skills demonstrated" table mapping features → job requirements, and a 5-minute
   demo path.

---

## 2. Proposed repository structure

```
invoice-iq/
├── README.md                  # recruiter-facing: arch diagram, skills map, demo
├── PLAN.md                    # this file
├── pyproject.toml             # deps (pinned), tool config (ruff, pytest, mypy)
├── requirements.txt           # generated/locked mirror for pip users
├── .env.example               # every env var, with safe placeholder values
├── .gitignore                 # .env, venv, __pycache__, data/, models/*.pt, etc.
├── .dockerignore
├── Dockerfile                 # production image for the FastAPI gateway
├── docker-compose.yml         # gateway + vector store (+ optional services) for local dev
├── Makefile                   # make install / test / lint / run / train / demo
│
├── .github/
│   └── workflows/
│       └── ci.yml             # lint (ruff) + type (mypy) + pytest on push/PR
│
├── config/
│   └── settings.py            # Pydantic-Settings: reads .env, toggles local vs GCP
│
├── src/
│   └── invoice_iq/
│       ├── __init__.py
│       ├── schemas/                  # ── Pydantic contracts (the spine) ──
│       │   ├── documents.py          #   DocumentType enum, RawDocument, OCRResult
│       │   ├── invoice.py            #   Invoice, LineItem, Vendor, Money, Money totals
│       │   └── api.py                #   request/response models for FastAPI
│       │
│       ├── ingestion/                # ── parse PDFs → text + layout ──
│       │   ├── base.py               #   OCRProvider protocol
│       │   ├── local_ocr.py          #   pdfplumber / PyMuPDF (+ Tesseract for scans)
│       │   └── docai.py              #   GCP Document AI impl (Phase 7, behind flag)
│       │
│       ├── extraction/               # ── OCR text → validated Invoice JSON ──
│       │   ├── base.py               #   Extractor protocol
│       │   ├── rule_based.py         #   regex/layout heuristics → Pydantic (local)
│       │   └── llm_extractor.py      #   LLM structured-output extraction (local/Vertex)
│       │
│       ├── classifier/               # ── the trained PyTorch model ──
│       │   ├── dataset.py            #   torch Dataset; synthetic data generator
│       │   ├── model.py              #   the nn.Module (text/feature classifier)
│       │   ├── train.py              #   training loop, checkpointing, metrics
│       │   ├── evaluate.py           #   confusion matrix, P/R/F1 on held-out set
│       │   └── predict.py            #   load checkpoint → DocumentType + confidence
│       │
│       ├── rag/                      # ── chunk → embed → store → retrieve → answer ──
│       │   ├── chunking.py           #   field-aware + text chunking
│       │   ├── embeddings.py         #   EmbeddingProvider protocol (local ST / Vertex)
│       │   ├── vector_store.py       #   VectorStore protocol; Chroma local impl
│       │   └── qa.py                 #   retrieval-augmented Q&A chain
│       │
│       ├── agent/                    # ── LangGraph orchestration ──
│       │   ├── state.py              #   typed AgentState (TypedDict/Pydantic)
│       │   ├── tools.py              #   classify/extract/retrieve/answer as tools
│       │   ├── graph.py              #   the StateGraph wiring + conditional edges
│       │   └── nodes.py              #   node fns: classify→extract→retrieve→answer→recommend
│       │
│       ├── serving/                  # ── FastAPI gateway ──
│       │   ├── app.py                #   FastAPI app factory, lifespan, routers
│       │   ├── routes.py             #   /classify /extract /ingest /ask /agent /health
│       │   ├── deps.py               #   DI: provider selection from settings
│       │   └── monitoring.py         #   /metrics, latency + prediction-distribution logging
│       │
│       └── vertex/                   # ── GCP glue (Phase 7+, all behind flags) ──
│           ├── deploy_classifier.py  #   upload model → Vertex endpoint
│           └── client.py             #   call Vertex endpoint from serving layer
│
├── data/                      # gitignored; synthetic PDFs + generated datasets
│   ├── samples/               # a few committed tiny sample invoices for the demo
│   └── generated/             # synthetic training corpus (gitignored)
│
├── models/                    # gitignored; trained .pt checkpoints + metrics.json
│
├── scripts/
│   ├── generate_synthetic_data.py    # make invoice/receipt/contract PDFs + labels
│   └── demo.py                       # one-command end-to-end demo
│
└── tests/
    ├── conftest.py
    ├── test_schemas.py
    ├── test_ingestion.py
    ├── test_extraction.py
    ├── test_classifier.py            # trains a tiny model on fixtures, asserts learning
    ├── test_rag.py
    ├── test_agent.py
    └── test_api.py                   # FastAPI TestClient over all endpoints
```

---

## 3. Tech stack (versions pinned at install time in each phase)

| Concern | Local (default) | GCP swap-in |
|---|---|---|
| PDF parse / OCR | `pdfplumber`, `PyMuPDF`, `pytesseract` | Document AI |
| Classifier | **PyTorch** (`torch`), `scikit-learn` (metrics) | Vertex AI Endpoint |
| Extraction LLM | Claude API or local rules | Vertex AI (Gemini) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Vertex AI text-embeddings |
| Vector store | `chromadb` (persistent, local) | Vertex AI Vector Search |
| Agent | `langgraph`, `langchain-core` | same (model behind it swaps) |
| API | `fastapi`, `uvicorn`, `pydantic` v2, `pydantic-settings` | Cloud Run |
| MLOps | `docker`, `docker-compose`, GitHub Actions, `ruff`, `mypy`, `pytest` | Cloud Build (optional) |

**Python:** 3.11 (Vertex/Cloud Run friendly). **Dependency management:** `pyproject.toml`
as source of truth + pinned `requirements.txt`. Virtualenv in `.venv/`.

---

## 4. Phased build plan (local-first → GCP)

Each phase = a vertical slice that ends **green**: passing tests, README updated,
one commit. I will run the tests and show you output before telling you what to commit.

### Phase 0 — Scaffold & guardrails
- Create repo skeleton, `pyproject.toml`, `.gitignore`, `.env.example`, `Makefile`.
- Set up `ruff` + `mypy` + `pytest` config; one trivial passing test to prove the harness.
- `git init`, first commit.
- **Exit:** `make test` green; CI config present (runs on next push).

### Phase 1 — Schemas (the spine)
- Pydantic v2 models: `DocumentType`, `RawDocument`, `OCRResult`, `LineItem`,
  `Vendor`, `Money`, `Invoice` (with validators: totals reconcile, dates parse,
  currency normalized).
- **Tests:** valid/invalid construction, total-reconciliation validator, JSON round-trip.
- **Exit:** `test_schemas.py` green; commit.

### Phase 2 — Ingestion + extraction (local)
- `OCRProvider` protocol + `local_ocr.py` (pdfplumber/PyMuPDF, Tesseract fallback).
- `Extractor` protocol + `rule_based.py` (regex/layout → `Invoice`) and an
  LLM-based `llm_extractor.py` for messy cases (local model or Claude API via env key).
- `scripts/generate_synthetic_data.py`: render realistic invoice/receipt/contract PDFs
  (reportlab) with ground-truth labels + field JSON.
- **Tests:** parse a committed sample PDF → assert extracted fields match ground truth.
- **Exit:** `test_ingestion.py`, `test_extraction.py` green; commit.

### Phase 3 — PyTorch classifier (genuinely trained)
- Synthetic corpus → `dataset.py` (torch `Dataset` over text/layout features).
- `model.py`: a real `nn.Module` (e.g. embedding-bag or small CNN/MLP over TF-IDF
  or learned token embeddings) classifying invoice / receipt / contract.
- `train.py`: training loop, train/val split, checkpoint to `models/`, write
  `metrics.json`. `evaluate.py`: confusion matrix + macro-F1 on held-out set.
- `predict.py`: load checkpoint → `(DocumentType, confidence)`.
- **Tests:** train a *tiny* model on fixtures inside the test and assert it learns
  (accuracy beats random by a clear margin); assert `predict` returns valid schema.
- **Exit:** `test_classifier.py` green; reported real metrics in README; commit.

### Phase 4 — RAG pipeline (local)
- `chunking.py` (field-aware + sliding-window text), `embeddings.py`
  (`sentence-transformers`), `vector_store.py` (Chroma persistent), `qa.py`
  (retrieve top-k → grounded answer with citations to source fields/chunks).
- **Tests:** ingest extracted invoices → ask "what's the total for vendor X?" →
  assert retrieval hits the right doc and answer contains the value.
- **Exit:** `test_rag.py` green; commit.

### Phase 5 — Agentic layer (LangGraph)
- Typed `AgentState`; tools wrapping classify/extract/retrieve/answer.
- `graph.py`: `StateGraph` with conditional edges:
  `classify → extract → embed/retrieve → answer → recommend_next_action`.
- Recommend-next-action node (e.g. "flag for human review", "approve", "ask vendor").
- **Tests:** run the graph on a sample PDF end-to-end with a fake/deterministic LLM;
  assert state transitions and final typed output.
- **Exit:** `test_agent.py` green; commit.

### Phase 6 — FastAPI serving + monitoring + MLOps (local)
- `app.py`/`routes.py`: `/health`, `/ingest`, `/classify`, `/extract`, `/ask`,
  `/agent`, `/metrics`. DI selects local providers from settings.
- `monitoring.py`: request latency, error rate, prediction-class distribution logging.
- `Dockerfile` (multi-stage, non-root), `docker-compose.yml` (gateway + Chroma).
- `.github/workflows/ci.yml`: ruff + mypy + pytest on push/PR.
- **Tests:** FastAPI `TestClient` covers every endpoint; container builds locally.
- **Exit:** `test_api.py` green; `docker compose up` serves the demo; commit.

> 🚦 **End of Phase 6 = a complete, impressive system with ZERO cloud spend.**
> Everything below is additive and gated on your explicit go-ahead + cost review.

### Phase 7 — GCP swap-ins (gated, cost-reviewed, opt-in)
For each item I will **stop first** and give you: what it does, the exact command,
the free-tier limit, the expected cost, and the local fallback that already works.
- **Document AI** invoice parser as an `OCRProvider`/`Extractor` impl behind a flag.
- **Vertex AI endpoint** for the PyTorch classifier (`deploy_classifier.py` + `client.py`).
- **Vertex embeddings / Vector Search** as alt `EmbeddingProvider`/`VectorStore`.
- **Cloud Run** deploy of the gateway (Dockerfile already built in Phase 6).
- **Tests:** provider-contract tests run against mocks in CI; live calls are manual.
- **Exit:** documented in README with screenshots; flags default to local.

### Phase 8 — Polish & recruiter packaging
- README architecture diagram, skills→requirements table, GIF/screenshots of demo,
  honest metrics, "how to run in 5 minutes", and a short "design decisions" section.
- Optional: simple Streamlit/HTML demo UI over the API.

---

## 5. Cost & safety policy (explicit)

- Nothing in Phases 0–6 touches GCP or any paid API by default. (If `llm_extractor`
  uses a hosted LLM, it's behind an env key and an off-by-default flag; rule-based
  extraction is the default path so tests never require a paid key.)
- Before **any** command that authenticates to GCP or can incur cost, I stop and tell
  you: purpose, exact command, free-tier coverage, estimated cost, and the local
  alternative. I do not run it without your "yes."
- All secrets via env vars + `.env` (gitignored). `.env.example` documents every var
  with placeholders. No credentials in code, tests, or commits.

---

## 6. Definition of done (per phase)

- [ ] Code typed (Pydantic + mypy-clean) and documented with docstrings.
- [ ] `pytest` green; I show you the run output.
- [ ] `ruff` clean.
- [ ] README section updated.
- [ ] I tell you the exact `git add` / `git commit -m "..."` to run.

---

## 7. Open questions for you before Phase 0

1. **Extraction LLM:** OK to use the **Claude API** (via `ANTHROPIC_API_KEY`) as the
   optional LLM-extraction path, with rule-based as the always-on default? Or keep it
   100% local (rules only) until Vertex/Gemini in Phase 7?
2. **GitHub:** do you want a remote repo created now (so CI runs on push), or stay
   local until later?
3. **Demo UI:** is the API + `scripts/demo.py` enough, or do you want a small
   Streamlit UI in Phase 8 for the recruiter screen-share?
4. **Classifier input:** train on **extracted text features** (fast, robust, fully
   local) — confirm that's the right altitude vs. a heavier layout-image model.

---

**Next step:** review this plan. On your go-ahead I'll start **Phase 0** (scaffold +
guardrails) only, and stop again with passing tests before any further code.
