"""
HuggingFace Datasets Loader
────────────────────────────
Integrates with the HuggingFace Hub REST API to search, browse, and extract
dataset metadata, then normalizes it into NDP/CKAN-compatible flat fields.

API endpoints used (no auth required for public datasets):
  Search:  GET https://huggingface.co/api/datasets?search=<q>&limit=N&full=true
  Detail:  GET https://huggingface.co/api/datasets/<owner>/<name>
  README:  GET https://huggingface.co/datasets/<owner>/<name>/resolve/main/README.md

Key fields returned by the HF API:
  id, author, sha, lastModified, private, disabled, gated,
  downloads, likes, tags, cardData (license, language, task_categories,
  pretty_name, size_categories, dataset_info), description (from README)
"""

import re
import json
import requests
from datetime import datetime
from typing import Optional, Callable

HF_API_BASE   = "https://huggingface.co/api"
HF_BASE       = "https://huggingface.co"
HF_DATASETS_URL = "https://huggingface.co/datasets"

REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "NDP-Ingestion-Agent/1.0 (research; ndp.sdsc.edu)",
}

# ── Mime-type helpers ─────────────────────────────────────────────────────────
FORMAT_MIME = {
    "parquet": "application/parquet",
    "csv":     "text/csv",
    "json":    "application/json",
    "jsonl":   "application/jsonlines",
    "txt":     "text/plain",
    "arrow":   "application/arrow",
    "zip":     "application/zip",
    "tar":     "application/x-tar",
    "audio":   "audio/flac",
    "image":   "image/jpeg",
    "video":   "video/mp4",
}

SIZE_CATEGORY_MAP = {
    "n<1K":    "< 1 K examples",
    "1K<n<10K":"1 K – 10 K examples",
    "10K<n<100K": "10 K – 100 K examples",
    "100K<n<1M":  "100 K – 1 M examples",
    "1M<n<10M":   "1 M – 10 M examples",
    "10M<n<100M": "10 M – 100 M examples",
    "100M<n<1B":  "100 M – 1 B examples",
    "n>1B":    "> 1 B examples",
}

# ── Network helpers ───────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, token: str = None,
         timeout: int = 20) -> Optional[requests.Response]:
    headers = dict(REQUEST_HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        return r
    except Exception:
        return None


def check_hf_connectivity(token: str = None) -> tuple[bool, str]:
    """Return (ok, message) — checks if HF API is reachable."""
    r = _get(f"{HF_API_BASE}/datasets", params={"limit": 1}, token=token)
    if r is None:
        return False, "Network error — cannot reach huggingface.co"
    if r.status_code == 200:
        return True, "Connected"
    if r.status_code == 401:
        return False, "Authentication required — provide a HF token"
    return False, f"HTTP {r.status_code}"


# ── Search ────────────────────────────────────────────────────────────────────

def search_hf_datasets(
    query: str = "",
    author: str = "",
    tags: list = None,
    limit: int = 10,
    sort: str = "downloads",     # downloads | likes | lastModified | trending
    token: str = None,
) -> list[dict]:
    """
    Search the HuggingFace Hub for datasets.

    Returns a list of lightweight dataset dicts (id, author, tags, downloads,
    likes, lastModified, license, size_categories).
    """
    params = {"limit": limit, "full": "true", "sort": sort, "direction": -1}
    if query:
        params["search"] = query
    if author:
        params["author"] = author
    if tags:
        params["filter"] = ",".join(tags)

    r = _get(f"{HF_API_BASE}/datasets", params=params, token=token)
    if r is None or not r.ok:
        return []

    try:
        items = r.json()
        return [_summarise_hf_item(it) for it in items if isinstance(it, dict)]
    except Exception:
        return []


def _summarise_hf_item(item: dict) -> dict:
    """Extract display-friendly summary from a raw HF API dataset item."""
    card = item.get("cardData") or {}
    tags = item.get("tags") or []
    # Extract license from tags (format: "license:mit") or cardData
    license_val = card.get("license", "")
    if not license_val:
        for t in tags:
            if t.startswith("license:"):
                license_val = t.split(":", 1)[1]
                break

    # Task categories
    tasks = card.get("task_categories") or []
    if not tasks:
        tasks = [t.split(":", 1)[1] for t in tags if t.startswith("task_categories:")]

    # Languages
    langs = card.get("language") or []
    if not langs:
        langs = [t.split(":", 1)[1] for t in tags if t.startswith("language:")]

    # Size
    sizes = card.get("size_categories") or []
    size_label = SIZE_CATEGORY_MAP.get(sizes[0], sizes[0]) if sizes else ""

    # Clean tags (remove prefix metadata tags, keep human-readable)
    clean_tags = [t for t in tags if ":" not in t]

    repo_id = item.get("id", "")
    return {
        "id": repo_id,
        "author": item.get("author", repo_id.split("/")[0] if "/" in repo_id else ""),
        "name": repo_id.split("/")[-1] if "/" in repo_id else repo_id,
        "pretty_name": card.get("pretty_name") or repo_id,
        "downloads": item.get("downloads", 0),
        "likes": item.get("likes", 0),
        "lastModified": item.get("lastModified", ""),
        "private": item.get("private", False),
        "gated": item.get("gated", False),
        "license": license_val,
        "tasks": tasks,
        "languages": langs,
        "size_label": size_label,
        "tags": clean_tags,
        "url": f"{HF_DATASETS_URL}/{repo_id}",
    }


# ── Full dataset detail ───────────────────────────────────────────────────────

def fetch_hf_dataset_detail(repo_id: str, token: str = None) -> Optional[dict]:
    """
    Fetch full metadata for a single HuggingFace dataset via the Hub API.
    Returns raw API response dict, or None on failure.
    """
    r = _get(f"{HF_API_BASE}/datasets/{repo_id}", token=token)
    if r and r.ok:
        try:
            return r.json()
        except Exception:
            pass
    return None


def fetch_hf_readme(repo_id: str, token: str = None) -> Optional[str]:
    """Fetch the README.md (dataset card) for a HuggingFace dataset."""
    urls = [
        f"{HF_BASE}/datasets/{repo_id}/resolve/main/README.md",
        f"{HF_BASE}/datasets/{repo_id}/raw/main/README.md",
    ]
    for url in urls:
        r = _get(url, token=token)
        if r and r.ok and len(r.text) > 50:
            return r.text
    return None


def _parse_readme(readme: str) -> dict:
    """
    Parse a HuggingFace dataset card (README.md with YAML front-matter).
    Extracts: description, license, language, task_categories, tags,
    pretty_name, size_categories, dataset_info fields.
    """
    result = {}

    # ── YAML front-matter ────────────────────────────────────────────────────
    yaml_match = re.match(r'^---\s*\n(.*?)\n---', readme, re.DOTALL)
    if yaml_match:
        yaml_block = yaml_match.group(1)

        def _yaml_list(key):
            m = re.search(rf'^{key}:\s*\n((?:\s*-\s*.+\n)+)', yaml_block,
                          re.MULTILINE | re.IGNORECASE)
            if m:
                return [re.sub(r'^\s*-\s*', '', ln).strip()
                        for ln in m.group(1).splitlines() if ln.strip()]
            # Inline list
            m2 = re.search(rf'^{key}:\s*\[([^\]]+)\]', yaml_block,
                           re.MULTILINE | re.IGNORECASE)
            if m2:
                return [x.strip().strip('"\'') for x in m2.group(1).split(',')]
            return []

        def _yaml_val(key):
            m = re.search(rf'^{key}:\s*(.+)', yaml_block,
                          re.MULTILINE | re.IGNORECASE)
            return m.group(1).strip().strip('"\'') if m else ""

        result["license"]          = _yaml_val("license")
        result["pretty_name"]      = _yaml_val("pretty_name")
        result["task_categories"]  = _yaml_list("task_categories")
        result["language"]         = _yaml_list("language")
        result["tags"]             = _yaml_list("tags")
        result["size_categories"]  = _yaml_list("size_categories")

    # ── Description: first substantive paragraph after front-matter ──────────
    body = re.sub(r'^---.*?---\s*\n', '', readme, flags=re.DOTALL)
    # Remove HTML comments, badges
    body = re.sub(r'<!--.*?-->', '', body, flags=re.DOTALL)
    body = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', body)  # badge links
    body = re.sub(r'^\s*#.*\n', '', body, flags=re.MULTILINE)  # headings
    paras = [p.strip() for p in body.split('\n\n') if len(p.strip()) > 60]
    if paras:
        result["description"] = paras[0]

    return result


# ── Full extraction pipeline ──────────────────────────────────────────────────

def extract_hf_dataset(
    repo_id: str,
    token: str = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Full extraction pipeline for one HuggingFace dataset.

    Layer 1: Hub API detail  → structured card metadata
    Layer 2: README.md       → description + YAML front-matter
    Layer 3: merge + fill defaults

    Returns a flat NDP metadata dict.
    """
    def log(m):
        if progress_cb:
            progress_cb(m)

    log(f"📡 Fetching Hub API for `{repo_id}`…")
    detail = fetch_hf_dataset_detail(repo_id, token=token)

    log("📄 Fetching README / dataset card…")
    readme_raw = fetch_hf_readme(repo_id, token=token)
    readme_parsed = _parse_readme(readme_raw) if readme_raw else {}

    return _normalize_hf_to_flat(repo_id, detail, readme_parsed, log)


def _normalize_hf_to_flat(
    repo_id: str,
    detail: Optional[dict],
    readme: dict,
    log: Callable,
) -> dict:
    """Merge API detail + README into a flat NDP metadata dict."""

    flat = {}
    card = (detail or {}).get("cardData") or {}
    tags_raw = (detail or {}).get("tags") or []

    # ── Title ────────────────────────────────────────────────────────────────
    pretty = card.get("pretty_name") or readme.get("pretty_name") or ""
    flat["title"] = pretty or repo_id.split("/")[-1].replace("-", " ").replace("_", " ").title()

    # ── Notes / Description ───────────────────────────────────────────────────
    desc = readme.get("description", "")
    if not desc:
        # Fall back to a generated description
        tasks = card.get("task_categories") or readme.get("task_categories") or []
        langs = card.get("language") or readme.get("language") or []
        desc_parts = [f"HuggingFace dataset `{repo_id}`."]
        if tasks:
            desc_parts.append(f"Task categories: {', '.join(tasks[:4])}.")
        if langs:
            desc_parts.append(f"Languages: {', '.join(langs[:4])}.")
        desc = " ".join(desc_parts)
    flat["notes"] = desc[:3000]

    # ── Tags ─────────────────────────────────────────────────────────────────
    # Combine README tags, task_categories, and clean hub tags
    task_tags = card.get("task_categories") or readme.get("task_categories") or []
    yaml_tags = readme.get("tags") or []
    hub_clean = [t for t in tags_raw if ":" not in t]
    all_tags = list(dict.fromkeys(task_tags + yaml_tags + hub_clean))  # dedupe, preserve order
    flat["tags"] = all_tags[:20] if all_tags else ["machine-learning"]

    # ── License ───────────────────────────────────────────────────────────────
    license_val = (card.get("license") or readme.get("license") or
                   next((t.split(":",1)[1] for t in tags_raw if t.startswith("license:")), ""))
    flat["extras.license"] = license_val

    # ── Dates ─────────────────────────────────────────────────────────────────
    last_mod = (detail or {}).get("lastModified", "")
    if last_mod:
        date_str = last_mod[:10]  # YYYY-MM-DD
        flat["extras.issueDate"]      = date_str
        flat["extras.lastUpdateDate"] = date_str

    # ── Contact / Author ──────────────────────────────────────────────────────
    author = (detail or {}).get("author", repo_id.split("/")[0] if "/" in repo_id else "")
    flat["extras.pocName"]  = author
    flat["extras.pocEmail"] = f"https://huggingface.co/{author}"   # best we can do without scraping

    # ── Upload type / data type ───────────────────────────────────────────────
    flat["extras.uploadType"] = "dataset"
    # Infer dataType from task categories
    tasks_lower = " ".join(task_tags).lower()
    if any(x in tasks_lower for x in ["image", "vision", "object"]):
        flat["extras.dataType"] = "imagery"
    elif any(x in tasks_lower for x in ["audio", "speech", "voice"]):
        flat["extras.dataType"] = "audio"
    elif any(x in tasks_lower for x in ["video"]):
        flat["extras.dataType"] = "video"
    elif any(x in tasks_lower for x in ["text", "nlp", "language", "translation",
                                         "classification", "generation", "qa"]):
        flat["extras.dataType"] = "text"
    elif any(x in tasks_lower for x in ["tabular", "regression", "forecasting"]):
        flat["extras.dataType"] = "tabular"
    else:
        flat["extras.dataType"] = "tabular"

    # ── Size info (extras) ────────────────────────────────────────────────────
    sizes = card.get("size_categories") or readme.get("size_categories") or []
    if sizes:
        flat["extras.sizeCategory"] = SIZE_CATEGORY_MAP.get(sizes[0], sizes[0])

    downloads = (detail or {}).get("downloads", 0)
    if downloads:
        flat["extras.downloads"] = str(downloads)

    likes = (detail or {}).get("likes", 0)
    if likes:
        flat["extras.likes"] = str(likes)

    langs = card.get("language") or readme.get("language") or []
    if langs:
        flat["extras.language"] = ", ".join(langs[:6])

    flat["extras.publisher"] = "HuggingFace Hub"

    # ── Resource ──────────────────────────────────────────────────────────────
    flat["resource.name"]        = flat["title"]
    flat["resource.description"] = f"HuggingFace dataset {repo_id} — access via the Hub API or datasets library."
    flat["resource.url"]         = f"{HF_DATASETS_URL}/{repo_id}"
    flat["resource.status"]      = "active"

    # Format + mimetype — infer from dataset_info or siblings
    siblings = (detail or {}).get("siblings") or []
    exts = set()
    for s in siblings:
        fname = s.get("rfilename", "")
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext in FORMAT_MIME:
            exts.add(ext)

    # Prefer parquet (HF auto-converts), then whatever else is present
    if "parquet" in exts or not exts:
        flat["resource.format"]   = "Parquet"
        flat["resource.mimetype"] = "application/parquet"
    else:
        ext = next(iter(exts))
        flat["resource.format"]   = ext.upper()
        flat["resource.mimetype"] = FORMAT_MIME.get(ext, "application/octet-stream")

    flat["source_format"]  = "HuggingFace Hub"
    flat["_source_url"]    = f"{HF_DATASETS_URL}/{repo_id}"
    flat["_hf_repo_id"]    = repo_id

    log(f"✅ Extraction complete — {len([v for v in flat.values() if v])} fields populated.")
    return flat


# ── Utility: repo_id from URL ─────────────────────────────────────────────────

def repo_id_from_url(url: str) -> Optional[str]:
    """
    Extract repo_id from a HuggingFace dataset URL.

    Examples:
      https://huggingface.co/datasets/nyu-mll/glue  →  nyu-mll/glue
      https://huggingface.co/datasets/squad          →  squad
    """
    m = re.search(r'huggingface\.co/datasets/([^/?#\s]+(?:/[^/?#\s]+)?)', url)
    return m.group(1) if m else None


# ── Curated popular datasets (offline fallback / discovery) ───────────────────
# Used when HF API is unreachable or for the "Browse popular" UI panel.

POPULAR_HF_DATASETS = [
    {
        "id": "nyu-mll/multi_nli",
        "pretty_name": "MultiNLI",
        "tasks": ["text-classification", "natural-language-inference"],
        "license": "cc-by-3.0",
        "languages": ["en"],
        "size_label": "100 K – 1 M examples",
        "description": "Multi-Genre Natural Language Inference corpus with 433k annotated sentence pairs across 10 genres.",
        "downloads": 2100000,
    },
    {
        "id": "squad",
        "pretty_name": "SQuAD",
        "tasks": ["question-answering", "extractive-qa"],
        "license": "cc-by-sa-4.0",
        "languages": ["en"],
        "size_label": "100 K – 1 M examples",
        "description": "Stanford Question Answering Dataset — 100k+ crowdsourced Q&A pairs on Wikipedia articles.",
        "downloads": 5800000,
    },
    {
        "id": "imdb",
        "pretty_name": "IMDB Reviews",
        "tasks": ["text-classification", "sentiment-analysis"],
        "license": "apache-2.0",
        "languages": ["en"],
        "size_label": "10 K – 100 K examples",
        "description": "Large Movie Review Dataset for binary sentiment classification — 50,000 reviews.",
        "downloads": 8200000,
    },
    {
        "id": "common_voice",
        "pretty_name": "Mozilla Common Voice",
        "tasks": ["automatic-speech-recognition"],
        "license": "cc0-1.0",
        "languages": ["multilingual"],
        "size_label": "1 M – 10 M examples",
        "description": "Open-source multi-language speech corpus from Mozilla with over 20,000 hours of recorded voice data.",
        "downloads": 3100000,
    },
    {
        "id": "wikimedia/wikipedia",
        "pretty_name": "Wikipedia",
        "tasks": ["text-generation", "language-modeling"],
        "license": "cc-by-sa-4.0",
        "languages": ["multilingual"],
        "size_label": "> 1 B examples",
        "description": "Wikipedia text dumps across 20+ languages, preprocessed for NLP tasks.",
        "downloads": 12000000,
    },
    {
        "id": "ai4bharat/samanantar",
        "pretty_name": "Samanantar",
        "tasks": ["translation"],
        "license": "cc0-1.0",
        "languages": ["hi", "bn", "ta", "te", "mr", "gu"],
        "size_label": "100 M – 1 B examples",
        "description": "Largest publicly available parallel corpus for Indic languages — 46M+ sentence pairs across 11 language pairs.",
        "downloads": 280000,
    },
    {
        "id": "code_search_net",
        "pretty_name": "CodeSearchNet",
        "tasks": ["code-retrieval", "text-to-code"],
        "license": "cc-by-sa-4.0",
        "languages": ["code"],
        "size_label": "1 M – 10 M examples",
        "description": "2M comment-code pairs from open-source libraries in 6 programming languages for code search research.",
        "downloads": 950000,
    },
    {
        "id": "cnn_dailymail",
        "pretty_name": "CNN / DailyMail",
        "tasks": ["summarization", "question-answering"],
        "license": "apache-2.0",
        "languages": ["en"],
        "size_label": "100 K – 1 M examples",
        "description": "News article summarization dataset with 300k+ article-highlight pairs from CNN and Daily Mail.",
        "downloads": 4400000,
    },
]


def get_popular_datasets_catalog() -> list[dict]:
    """Return the curated popular HF datasets list (no network required)."""
    return POPULAR_HF_DATASETS


def popular_dataset_to_flat(entry: dict) -> dict:
    """Convert a popular-catalog entry to a flat NDP metadata dict."""
    repo_id = entry["id"]
    flat = {}
    flat["title"]       = entry.get("pretty_name") or repo_id.split("/")[-1]
    flat["notes"]       = entry.get("description", "")
    flat["tags"]        = entry.get("tasks", []) + [entry.get("license", "")]
    flat["tags"]        = [t for t in flat["tags"] if t]

    flat["extras.uploadType"]   = "dataset"
    flat["extras.dataType"]     = _infer_data_type(entry.get("tasks", []))
    flat["extras.license"]      = entry.get("license", "")
    flat["extras.language"]     = ", ".join(entry.get("languages", []))
    flat["extras.sizeCategory"] = entry.get("size_label", "")
    flat["extras.downloads"]    = str(entry.get("downloads", ""))
    flat["extras.publisher"]    = "HuggingFace Hub"
    flat["extras.pocName"]      = repo_id.split("/")[0] if "/" in repo_id else "HuggingFace"
    flat["extras.pocEmail"]     = f"https://huggingface.co/{flat['extras.pocName']}"

    flat["resource.name"]        = flat["title"]
    flat["resource.description"] = flat["notes"][:200]
    flat["resource.url"]         = f"{HF_DATASETS_URL}/{repo_id}"
    flat["resource.format"]      = "Parquet"
    flat["resource.mimetype"]    = "application/parquet"
    flat["resource.status"]      = "active"

    flat["source_format"] = "HuggingFace (catalog)"
    flat["_source_url"]   = f"{HF_DATASETS_URL}/{repo_id}"
    flat["_hf_repo_id"]   = repo_id
    return flat


def _infer_data_type(tasks: list) -> str:
    t = " ".join(tasks).lower()
    if any(x in t for x in ["image", "vision", "object-detection"]):
        return "imagery"
    if any(x in t for x in ["audio", "speech", "asr"]):
        return "audio"
    if any(x in t for x in ["video"]):
        return "video"
    if any(x in t for x in ["tabular", "regression", "forecasting"]):
        return "tabular"
    return "text"
