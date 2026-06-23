# NDP Intelligent Data Ingestion Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-red?style=flat-square&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/LLM-12%20models-green?style=flat-square" alt="LLMs">
  <img src="https://img.shields.io/badge/NDP-CKAN%20Compatible-orange?style=flat-square" alt="NDP">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" alt="License">
</p>

> **DSE 260-B Capstone · MAS in Data Science & Engineering (Cohort 11, Group 2)**  
> **Team:** Chung Loh · Ritesh Saxena  
> **Advisors:** Ilkay Altintas, Ph.D. · Taina Coleman, Ph.D.  
> **Institution:** UC San Diego — San Diego Supercomputer Center (SDSC)

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Data Sources](#data-sources)
- [LLM Models](#llm-models)
- [NDP Field Schema](#ndp-field-schema)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Project Structure](#project-structure)
- [Evaluation System](#evaluation-system)
- [Workflow Tabs](#workflow-tabs)
- [API Reference](#api-reference)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)

---

## Overview

The **NDP Intelligent Data Ingestion Agent** automates the process of transforming heterogeneous scientific dataset metadata into validated, CKAN-compatible records for the [National Data Platform (NDP)](https://nationaldataplatform.org) catalog.

Domain scientists often struggle to publish datasets on NDP because metadata normalization — mapping fields from ISO XML, EML, plain text, PDFs, or third-party catalogs like HuggingFace or MLCommons into the 14 required NDP CKAN fields — requires deep technical knowledge. This agent eliminates that barrier.

The system ingests raw metadata from five source types, extracts fields using format-specific loaders, uses an LLM to fill gaps and repair errors, validates the result against the live NDP API, and produces a submission-ready CKAN JSON package.

**Key results from evaluation across 8 ground-truth fixtures:**

| Metric | Score |
|--------|-------|
| Field completeness | 100% |
| Exact-match accuracy | 88.7% |
| Overall weighted score | 93.5 / 100 |
| Average loader time | < 0.5 ms |
| Preflight pass rate (post-enrichment) | 100% |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User (Streamlit UI)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │  input: file / URL / text / slug
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Loader Layer                            │
│  xml_loader  │  text_loader  │  pdf_loader                  │
│  mlcommons_loader  │  huggingface_loader                     │
└──────────────────────────┬──────────────────────────────────┘
                           │  flat metadata dict
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM Normalization & Enrichment                  │
│  llm_registry (12 models across 4 provider groups)          │
│  llm_provider: normalize → repair → human Q&A loop          │
└──────────────────────────┬──────────────────────────────────┘
                           │  enriched flat dict
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Validation Layer                          │
│  preflight_check (local)  →  NDP API validate endpoint      │
│  schema.py: 14 required fields · format rules               │
└──────────────────────────┬──────────────────────────────────┘
                           │  valid CKAN JSON
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Submission & Export                         │
│  Download JSON  ·  curl command  ·  CKAN API POST           │
└─────────────────────────────────────────────────────────────┘
```

The pipeline is **human-orchestrated**: Python drives each stage and hands off to the LLM at defined points. The loaders and validator are shaped like MCP tools and could be registered with an LLM tool-calling framework (e.g., Ollama's llama3.1+ tool API) to make the agent fully LLM-orchestrated in a future version.

---

## Features

- **Multi-source ingestion** — ISO 19139 XML, EML, plain text, PDF, NDP CKAN catalog, MLCommons, HuggingFace Hub
- **Deep CKAN extraction** — 4-layer pipeline: `package_show` API → ISO XML endpoint → resource URL scanning → notes heuristics
- **12 LLM models** across 4 provider groups, switchable at runtime from the sidebar
- **Human-in-the-loop Q&A** — agent asks targeted questions for each missing required field
- **LLM auto-repair** — sends validation errors back to the LLM for automatic JSON correction
- **Preflight + NDP API validation** — local field checks before hitting the live endpoint
- **System evaluation tab** — accuracy, timing, and token cost analysis across 8 ground-truth fixtures
- **JSON + CSV export** of evaluation results
- **One-click submission** to NDP CKAN catalog via API key

---

## Data Sources

### NDP CKAN Catalog

| Organization | Format | Records | URL |
|---|---|---|---|
| `bco_weather` | ISO/XML | 170 | https://ndp-test.sdsc.edu/catalog/organization/bco_weather |
| `clm_test` | TXT | 188 | https://ndp-test.sdsc.edu/catalog/organization/clm_test |
| `wfsi` | EML | 100 | https://ndp-test.sdsc.edu/catalog/organization/wfsi |

The deep-extraction pipeline fetches each package via `package_show`, then attempts the ISO XML endpoint (`/dataset/{name}/iso`), then scans resource URLs for embedded XML/EML/TXT metadata files, then falls back to parsing the `notes` field as plain text.

### Third-Party Sources

| Source | Access | Loader |
|---|---|---|
| **MLCommons** (`mlcommons.org/datasets`) | Live scrape + built-in catalog (5 datasets) | `mlcommons_loader.py` |
| **HuggingFace Hub** (`huggingface.co/datasets`) | REST API (`/api/datasets`) + README.md scrape + curated catalog (8 datasets) | `huggingface_loader.py` |
| **Wildfire PDFs** (`wildfiretaskforce.org/treatment-dashboard`) | File upload | `pdf_loader.py` |
| **Uploaded files** | PDF, XML, TXT | `pdf_loader.py`, `xml_loader.py`, `text_loader.py` |
| **Pasted text** | ISO XML, EML, plain text | auto-detected |

---

## LLM Models

12 models are registered across 4 provider groups. Switch between them from the sidebar at runtime — all use the same prompts.

### Group A — Ollama (local, free)

| Model key | Model | Endpoint |
|---|---|---|
| `ollama/llama3` | llama3 | `http://localhost:11434` |
| `ollama/llama2` | llama2 | `http://localhost:11434` |
| `ollama/mistral` | mistral | `http://localhost:11434` |

Requires [Ollama](https://ollama.com) running locally. No API key needed.

### Group B — NRP-ELLM / Nautilus (shared `ELLM_API_KEY`)

| Model key | Model |
|---|---|
| `ellm/gemma` | gemma |
| `ellm/qwen3` | qwen3 |
| `ellm/glm-v` | glm-v |
| `ellm/gpt-oss` | gpt-oss |
| `ellm/kimi` | kimi |
| `ellm/glm-4.7` | glm-4.7 |
| `ellm/minimax-m2` | minimax-m2 |

Endpoint: `https://ellm.nrp-nautilus.io/v1` (OpenAI-compatible).  
All 7 models share one key: set `ELLM_API_KEY` in `.env`.

### Group C — Cloud APIs

| Model key | Provider | Model | Key env-var |
|---|---|---|---|
| `anthropic/claude-sonnet` | Anthropic | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` |
| `openai/gpt-4o-mini` | OpenAI | gpt-4o-mini | `OPENAI_API_KEY` |

---

## NDP Field Schema

### Required fields — public datasets (14)

| Section | Field | CKAN key | Type |
|---|---|---|---|
| General | `title` | `title` | text |
| General | `notes` | `notes` | text |
| General | `tags` | `tags` | list (≥ 1) |
| General | `extras.uploadType` | `extras:uploadType` | `dataset` \| `service` \| `model` \| `collection` |
| General | `extras.dataType` | `extras:dataType` | text |
| Contact | `extras.pocName` | `extras:pocName` | text |
| Contact | `extras.pocEmail` | `extras:pocEmail` | text |
| Metadata | `extras.issueDate` | `extras:issueDate` | `YYYY-MM-DD` |
| Metadata | `extras.lastUpdateDate` | `extras:lastUpdateDate` | `YYYY-MM-DD` |
| Resource | `resource.name` | `resource:name` | text |
| Resource | `resource.description` | `resource:description` | text |
| Resource | `resource.mimetype` | `resource:mimetype` | MIME type |
| Resource | `resource.format` | *(resource body)* | text |
| Resource | `resource.status` | `resource:status` | `active` \| `archived` \| `deprecated` |

### Required fields — private datasets (7)

`title`, `notes`, `extras.uploadType`, `extras.issueDate`, `extras.lastUpdateDate`, `resource.name`, `resource.description`

### Enrichment fields (extracted, not NDP-required)

`resource.url`, `extras.license`, `extras.publisher`, `extras.language`, `extras.sizeCategory`, `extras.downloads`, `extras.size`

---

## Installation

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com) (for local models — optional if using cloud APIs)

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/ndp-ingestion-agent.git
cd ndp-ingestion-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
streamlit>=1.32.0
requests>=2.31.0
pypdf>=4.0.0
pdfplumber>=0.10.0
lxml>=5.0.0
pandas>=2.0.0
xmltodict>=0.13.0
python-dotenv>=1.0.0
pyyaml>=6.0
```

### 4. Install and start Ollama (for local models)

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull models (at least one required for local use)
ollama pull llama3
ollama pull mistral   # optional
ollama pull llama2    # optional

# Start the Ollama server
ollama serve
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

`.env` contents:

```env
# NRP / ELLM — shared key for all 7 NRP-Nautilus models
ELLM_API_KEY=your-ellm-api-key-here

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# OpenAI
OPENAI_API_KEY=your-openai-api-key-here

# Ollama (optional — defaults to http://localhost:11434)
# OLLAMA_BASE_URL=http://localhost:11434
```

**Credential resolution order:**
1. Sidebar input at runtime (stored in Streamlit session state)
2. `.env` file (loaded automatically on startup)
3. Shell environment variables

You only need keys for the provider groups you intend to use. Ollama requires no key.

---

## Running the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

To run the evaluation engine from the command line:

```bash
python eval/eval_engine.py
```

---

## Project Structure

```
ndp-ingestion-agent/
│
├── app.py                        # Main Streamlit application (6 tabs)
├── requirements.txt
├── .env.example                  # Credential template
├── README.md
│
├── loaders/                      # Source-specific extraction modules
│   ├── __init__.py
│   ├── xml_loader.py             # ISO 19139 XML + EML + deep CKAN extraction
│   ├── text_loader.py            # Plain-text heuristic parser (regex)
│   ├── pdf_loader.py             # PDF text extraction (pdfplumber / pypdf)
│   ├── mlcommons_loader.py       # MLCommons catalog + live page scraper
│   └── huggingface_loader.py     # HuggingFace Hub REST API + README parser
│
├── utils/                        # Shared utilities
│   ├── __init__.py
│   ├── schema.py                 # NDP/CKAN field definitions and validation rules
│   ├── validator.py              # Preflight check + NDP API validation
│   ├── llm_registry.py           # Unified model registry (12 models, 4 groups)
│   └── llm_provider.py           # LLM call interface: normalize, repair, questions
│
└── eval/                         # Evaluation system
    ├── __init__.py
    ├── eval_engine.py            # Accuracy + timing + token scoring engine
    └── eval_page.py              # Streamlit evaluation tab renderer
```

---

## Evaluation System

Run from the **📊 Evaluation** tab in the app or from the command line.

### Three evaluation dimensions

**1. Accuracy**

8 ground-truth fixtures (3 TXT · 2 MLCommons · 2 HuggingFace · 1 ISO/XML). Each fixture defines an input and a dict of expected field values. After the loader runs, every field is scored on a 5-tier scale:

| Score | Condition |
|-------|-----------|
| 1.0 | Exact string match + valid format |
| 0.6 | Partial match (≥ 60% word overlap) + valid format |
| 0.4 | Partial match, invalid format |
| 0.3 | Present, no match, valid format |
| 0.1 | Present, no match, invalid format |
| 0.0 | Missing |

Format validation checks: email syntax, `YYYY-MM-DD` date regex, allowed upload types (`dataset` / `service` / `model` / `collection`), MIME type structure (`type/subtype`), and allowed status values.

Aggregate metrics per fixture: `completeness_pct` (fields found / required), `accuracy_pct` (exact matches / required), `overall_score` (avg field score × 100).

**2. Timing**

Wall-clock time for the loader only, measured with `time.perf_counter()`. Excludes LLM call time, HTTP fetches, and Streamlit overhead. All loaders operate on in-memory data (regex, ElementTree, dict lookup) and run in under 2 ms.

LLM timing is model- and hardware-dependent: Ollama llama3 averages 3–8 s per call on a modern laptop; cloud APIs average 1–4 s.

**3. Token consumption**

Simulates the 3 LLM calls made per dataset (normalize → repair → questions) with realistic prompt and response sizes. Token counts use a character-ratio approximation (prose: 4.0 chars/token, JSON: 3.5 chars/token, calibrated against cl100k_base).

Cost formula per call:
```
cost = (prompt_tokens / 1000 × input_rate) + (response_tokens / 1000 × output_rate)
```

| Provider | Cost per extraction cycle |
|---|---|
| Ollama (local) | Free |
| NRP-ELLM (gemma, cheapest) | ~$0.000213 |
| OpenAI GPT-4o-mini | ~$0.000264 |
| Anthropic Claude Sonnet | ~$0.005970 |

Results can be exported as JSON (full field-level detail) or CSV from the evaluation tab.

---

## Workflow Tabs

| Tab | Purpose |
|---|---|
| **📥 1. Ingest Metadata** | Choose source: upload file, NDP CKAN URL, MLCommons, HuggingFace, paste text, or custom URL |
| **✏️ 2. Review & Enrich** | Edit fields manually, auto-fill missing fields with LLM, answer agent Q&A |
| **✅ 3. Validate** | Preflight check + NDP API validation + LLM auto-repair |
| **🚀 4. Submit** | Download JSON, view curl command, or submit directly via CKAN API |
| **🔭 5. Explore NDP** | Browse bco_weather, wfsi, wildfire PDFs, MLCommons, HuggingFace |
| **📊 6. Evaluation** | Run system evaluation: accuracy, timing, token cost across all models |

---

## API Reference

### NDP Validation Endpoint

```
POST https://ndp-test.sdsc.edu/catalog2/ndp/package_validate
Content-Type: application/json
```

Sample request:

```bash
curl -X POST https://ndp-test.sdsc.edu/catalog2/ndp/package_validate \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-dataset",
    "title": "My Dataset",
    "notes": "A detailed description of my dataset.",
    "owner_org": "my-org",
    "extras": [
      {"key": "uploadType", "value": "dataset"},
      {"key": "dataType",   "value": "tabular"},
      {"key": "pocName",    "value": "Dr. Jane Smith"},
      {"key": "pocEmail",   "value": "jsmith@ucsd.edu"},
      {"key": "issueDate",  "value": "2024-01-15"},
      {"key": "lastUpdateDate", "value": "2024-06-01"}
    ],
    "tags": [{"name": "climate"}, {"name": "sensors"}],
    "resources": [{
      "name": "Primary dataset file",
      "description": "CSV file with hourly measurements",
      "format": "CSV",
      "mimetype": "text/csv",
      "url": "https://example.com/data.csv",
      "status": "active"
    }]
  }'
```

### CKAN Package Create (submission)

```
POST https://ndp-test.sdsc.edu/catalog/api/3/action/package_create
X-CKAN-API-Key: <your-api-key>
Content-Type: application/json
```

### HuggingFace Hub API

```
GET https://huggingface.co/api/datasets?search=<query>&limit=10&sort=downloads
GET https://huggingface.co/api/datasets/<owner>/<name>
GET https://huggingface.co/datasets/<owner>/<name>/resolve/main/README.md
```

---

## Known Limitations

**MCP not yet implemented.** The project reports describe MCP (Model Context Protocol) as the bridge between the LLM and the extraction/validation tools. The current implementation is human-orchestrated — Python calls the loaders and validator directly and passes results to the LLM. The loaders and validator are already shaped like MCP tools. Connecting them via Ollama's tool-calling API (llama3.1+) would realize the original architecture.

**schema.py inconsistency.** `PUBLIC_OPTIONAL_FIELDS` currently lists fields that also appear in `PUBLIC_REQUIRED_FIELDS`. This variable should be removed or corrected; the spreadsheet and NDP validation endpoint are the authoritative source, and all 14 fields in `PUBLIC_REQUIRED_FIELDS` are required for public datasets.

**Live NDP CKAN access.** The NDP test catalog (`ndp-test.sdsc.edu`) requires network access. Firewall-restricted environments will see HTTP 403 errors on the deep-extraction and validation calls; the app falls back gracefully to CKAN API fields only.

**LLM accuracy not evaluated.** The evaluation system tests deterministic loaders only. LLM normalization accuracy depends on the active model, temperature, and prompt. Adding fixture-level LLM evaluation requires live model calls during testing.

**Token estimation is approximate.** The character-ratio estimator (4.0 chars/token) is accurate to ±15% for English prose. Exact counts require the `tiktoken` BPE tokenizer, which needs a one-time network download of the vocabulary file.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests to `eval/eval_engine.py` if adding a new loader
4. Run the evaluation: `python eval/eval_engine.py`
5. Commit: `git commit -m "feat: description of change"`
6. Push and open a pull request

### Adding a new loader

1. Create `loaders/your_loader.py` with a function that returns a flat `dict` with NDP field keys
2. Add ground-truth fixtures to `eval/eval_engine.py → GROUND_TRUTH_FIXTURES`
3. Wire the new source into the Ingest tab in `app.py`
4. Import the loader in `app.py` and the new Explore sub-tab in Tab 5

### Adding a new LLM model

Add an entry to `MODEL_REGISTRY` in `utils/llm_registry.py`:

```python
{
    "key":         "group/model-name",
    "label":       "Human Label (Provider)",
    "group":       "ollama" | "ellm" | "anthropic" | "openai",
    "model":       "api-model-name",
    "base_url":    "https://api.example.com/v1",
    "api_key_env": "MY_API_KEY_ENV_VAR",
    "notes":       "Short description for sidebar.",
}
```

If the provider uses an OpenAI-compatible `/v1/chat/completions` endpoint, set `"group": "ellm"` (it shares that backend). Otherwise add a new `_generate_*` backend function in `llm_registry.py`.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built for the <a href="https://nationaldataplatform.org">National Data Platform</a> ·
  UC San Diego SDSC · DSE 260-B Capstone 2025
</p>
