"""
MLCommons Loader: Scrapes and normalizes dataset metadata from mlcommons.org/datasets/
into NDP/CKAN-compatible flat metadata format.

Strategy:
  1. Known dataset catalog (hard-coded from site nav — always available, no network needed)
  2. Live page scraping per dataset (fetches the real page when network is available)
  3. LLM fallback for enrichment of scraped text

MLCommons datasets as of 2025:
  - Cognata
  - Dollar Street
  - Multilingual Spoken Words
  - People's Speech
  - Unsupervised People's Speech
"""

import re
import requests
from typing import Optional
from datetime import datetime

# ── Known catalog (always available, no scraping required) ────────────────────
# Populated from the public MLCommons website. Each entry is a superset of NDP fields.
MLCOMMONS_CATALOG = [
    {
        "slug": "peoples-speech",
        "title": "People's Speech",
        "url": "https://mlcommons.org/datasets/peoples-speech/",
        "notes": (
            "People's Speech is one of the world's largest English speech recognition "
            "datasets licensed for academic and commercial usage under CC-BY-SA 4.0. "
            "It includes 30,000+ hours of transcribed speech in English from a variety "
            "of speakers, recording conditions, and topics, designed to enable the "
            "training of large-vocabulary continuous speech recognition systems."
        ),
        "tags": ["speech", "audio", "ASR", "English", "transcription", "NLP", "deep learning"],
        "extras.uploadType": "dataset",
        "extras.dataType": "audio",
        "extras.pocName": "MLCommons",
        "extras.pocEmail": "datasets@mlcommons.org",
        "extras.issueDate": "2021-06-01",
        "extras.lastUpdateDate": "2021-06-01",
        "extras.license": "CC-BY-SA 4.0",
        "extras.publisher": "MLCommons",
        "extras.size": "~87,000 hours raw / ~30,000 hours transcribed",
        "extras.format": "FLAC / JSON",
        "resource.name": "People's Speech Dataset",
        "resource.description": "Audio files and transcriptions for large-scale English ASR training.",
        "resource.mimetype": "audio/flac",
        "resource.format": "FLAC",
        "resource.status": "active",
        "resource.url": "https://mlcommons.org/datasets/peoples-speech/",
        "resource.download": "https://huggingface.co/datasets/MLCommons/peoples_speech",
    },
    {
        "slug": "dollar-street",
        "title": "Dollar Street",
        "url": "https://mlcommons.org/datasets/dollar-street/",
        "notes": (
            "The MLCommons Dollar Street dataset is a collection of images of everyday "
            "household items from homes around the world, visually capturing socioeconomic "
            "diversity of traditionally underrepresented populations. It includes 38,479 images "
            "collected from 63 different countries tagged across 289 possible topics. "
            "Metadata includes region, country, and total household monthly income. "
            "Licensed under CC-BY and CC-BY-SA 4.0 for academic and commercial use."
        ),
        "tags": ["images", "computer vision", "diversity", "socioeconomic", "households", "global"],
        "extras.uploadType": "dataset",
        "extras.dataType": "imagery",
        "extras.pocName": "MLCommons",
        "extras.pocEmail": "datasets@mlcommons.org",
        "extras.issueDate": "2021-11-09",
        "extras.lastUpdateDate": "2021-11-09",
        "extras.license": "CC-BY / CC-BY-SA 4.0",
        "extras.publisher": "MLCommons",
        "extras.size": "101.3 GB",
        "extras.examples": "38,479 images",
        "extras.format": "JPG, PNG",
        "resource.name": "Dollar Street Dataset",
        "resource.description": "Images from 63 countries tagged across 289 household topics with socioeconomic metadata.",
        "resource.mimetype": "image/jpeg",
        "resource.format": "JPG",
        "resource.status": "active",
        "resource.url": "https://mlcommons.org/datasets/dollar-street/",
        "resource.download": "https://www.kaggle.com/datasets/mlcommons/the-dollar-street-dataset",
    },
    {
        "slug": "multilingual-spoken-words",
        "title": "Multilingual Spoken Words",
        "url": "https://mlcommons.org/datasets/multilingual-spoken-words/",
        "notes": (
            "The Multilingual Spoken Words Corpus (MSWC) is a large and growing audio dataset "
            "of spoken words in 50 languages, totalling 340,000+ keywords and more than "
            "23.4 million 1-second audio samples. It is designed for training and evaluating "
            "keyword spotting and spoken language understanding models across a broad range "
            "of languages. Licensed under CC-BY 4.0."
        ),
        "tags": ["speech", "audio", "multilingual", "keyword spotting", "NLP", "50 languages", "ASR"],
        "extras.uploadType": "dataset",
        "extras.dataType": "audio",
        "extras.pocName": "MLCommons",
        "extras.pocEmail": "datasets@mlcommons.org",
        "extras.issueDate": "2021-08-01",
        "extras.lastUpdateDate": "2021-08-01",
        "extras.license": "CC-BY 4.0",
        "extras.publisher": "MLCommons",
        "extras.size": "~124 GB",
        "extras.examples": "23.4 million audio samples",
        "extras.format": "OPUS",
        "resource.name": "Multilingual Spoken Words Corpus",
        "resource.description": "340,000+ keywords across 50 languages with 23.4M 1-second audio clips.",
        "resource.mimetype": "audio/ogg",
        "resource.format": "OPUS",
        "resource.status": "active",
        "resource.url": "https://mlcommons.org/datasets/multilingual-spoken-words/",
        "resource.download": "https://huggingface.co/datasets/MLCommons/multilingual_librispeech",
    },
    {
        "slug": "cognata",
        "title": "Cognata",
        "url": "https://mlcommons.org/datasets/cognata/",
        "notes": (
            "The Cognata dataset is a large-scale synthetic autonomous driving dataset "
            "generated by Cognata's simulation platform. It provides photorealistic "
            "synthetic video, LiDAR, radar, and semantic segmentation data for training "
            "and validating perception models in autonomous vehicles across diverse "
            "road conditions, environments, and weather scenarios."
        ),
        "tags": ["autonomous driving", "synthetic", "simulation", "LiDAR", "computer vision",
                 "perception", "video", "radar"],
        "extras.uploadType": "dataset",
        "extras.dataType": "imagery",
        "extras.pocName": "MLCommons",
        "extras.pocEmail": "datasets@mlcommons.org",
        "extras.issueDate": "2021-01-01",
        "extras.lastUpdateDate": "2022-01-01",
        "extras.license": "MLCommons License",
        "extras.publisher": "MLCommons / Cognata",
        "extras.format": "Video, LiDAR, Radar",
        "resource.name": "Cognata Synthetic Driving Dataset",
        "resource.description": "Synthetic photorealistic AV dataset with video, LiDAR, radar, and semantic labels.",
        "resource.mimetype": "video/mp4",
        "resource.format": "Video",
        "resource.status": "active",
        "resource.url": "https://mlcommons.org/datasets/cognata/",
        "resource.download": "https://mlcommons.org/datasets/cognata/",
    },
    {
        "slug": "unsupervised-peoples-speech",
        "title": "Unsupervised People's Speech",
        "url": "https://mlcommons.org/datasets/unsupervised-peoples-speech/",
        "notes": (
            "Unsupervised People's Speech is a companion dataset to People's Speech, "
            "containing raw (untranscribed) audio data. It provides a massive corpus "
            "for unsupervised and self-supervised speech representation learning. "
            "Includes diverse speakers, recording environments, and topics in English. "
            "Licensed for academic and commercial usage under CC-BY-SA 4.0."
        ),
        "tags": ["speech", "audio", "unsupervised learning", "self-supervised", "English",
                 "NLP", "speech representation"],
        "extras.uploadType": "dataset",
        "extras.dataType": "audio",
        "extras.pocName": "MLCommons",
        "extras.pocEmail": "datasets@mlcommons.org",
        "extras.issueDate": "2022-01-01",
        "extras.lastUpdateDate": "2022-01-01",
        "extras.license": "CC-BY-SA 4.0",
        "extras.publisher": "MLCommons",
        "extras.format": "FLAC",
        "resource.name": "Unsupervised People's Speech Dataset",
        "resource.description": "Raw untranscribed audio for unsupervised and self-supervised speech learning.",
        "resource.mimetype": "audio/flac",
        "resource.format": "FLAC",
        "resource.status": "active",
        "resource.url": "https://mlcommons.org/datasets/unsupervised-peoples-speech/",
        "resource.download": "https://huggingface.co/datasets/MLCommons/peoples_speech",
    },
]

MLCOMMONS_BASE = "https://mlcommons.org/datasets/"

# ── Scraping helpers ──────────────────────────────────────────────────────────

def _fetch_page(url: str, timeout: int = 20) -> Optional[str]:
    """Fetch raw HTML from a URL."""
    try:
        resp = requests.get(
            url, timeout=timeout,
            headers={"User-Agent": "NDP-Ingestion-Agent/1.0 (research)"}
        )
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def _parse_mlcommons_page(html: str, slug: str, url: str) -> dict:
    """
    Parse an MLCommons dataset page HTML into a flat metadata dict.
    Works without BeautifulSoup by using targeted regex on the page body.
    """
    # Strip script/style blocks
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

    # Extract raw text
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()

    result = {}

    # Title — look for h1
    h1 = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.IGNORECASE | re.DOTALL)
    if h1:
        title = re.sub(r'<[^>]+>', '', h1.group(1)).strip()
        if title:
            result['title'] = title

    # Main description — text after the h1, up to "About the dataset" or "Details"
    desc_match = re.search(
        r'</h1>\s*<p>(.*?)</p>',
        html, re.IGNORECASE | re.DOTALL
    )
    if desc_match:
        desc = re.sub(r'<[^>]+>', '', desc_match.group(1)).strip()
        if desc and len(desc) > 30:
            result['notes'] = desc

    # About section — grab longer paragraphs
    about_match = re.search(
        r'About the dataset.*?<p>(.*?)</p>',
        html, re.IGNORECASE | re.DOTALL
    )
    if about_match:
        about = re.sub(r'<[^>]+>', '', about_match.group(1)).strip()
        if about and len(about) > 50:
            existing = result.get('notes', '')
            result['notes'] = (existing + ' ' + about).strip() if existing else about

    # Details block — Date, Size, Format, Examples
    date_m = re.search(r'Date[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4})', text, re.IGNORECASE)
    if date_m:
        raw_date = date_m.group(1).replace('/', '-')
        # Normalize to YYYY-MM-DD
        parts = raw_date.split('-')
        if len(parts) == 3:
            result['extras.issueDate'] = raw_date
            result['extras.lastUpdateDate'] = raw_date

    size_m = re.search(r'Size[:\s]+([\d.,]+\s*(?:GB|MB|TB|KB)[^\n]*)', text, re.IGNORECASE)
    if size_m:
        result['extras.size'] = size_m.group(1).strip()

    fmt_m = re.search(r'Format[:\s]+([^\n.]+)', text, re.IGNORECASE)
    if fmt_m:
        fmt_val = fmt_m.group(1).strip()[:60]
        result['extras.format'] = fmt_val
        # Map to mimetype
        fmt_upper = fmt_val.upper()
        mime_map = {
            'FLAC': 'audio/flac',
            'OPUS': 'audio/ogg',
            'JPG': 'image/jpeg', 'JPEG': 'image/jpeg', 'PNG': 'image/png',
            'MP4': 'video/mp4', 'VIDEO': 'video/mp4',
            'CSV': 'text/csv', 'JSON': 'application/json',
            'ZIP': 'application/zip', 'TAR': 'application/x-tar',
        }
        for k, v in mime_map.items():
            if k in fmt_upper:
                result['resource.mimetype'] = v
                result['resource.format'] = k
                break

    examples_m = re.search(r'Examples[:\s]+([\d,]+)', text, re.IGNORECASE)
    if examples_m:
        result['extras.examples'] = examples_m.group(1).strip()

    # License
    license_m = re.search(r'(CC[-\s]BY(?:[-\s]SA)?(?:\s+\d+\.\d+)?|Apache[-\s]\d+\.\d+|MIT License|MLCommons License)',
                           text, re.IGNORECASE)
    if license_m:
        result['extras.license'] = license_m.group(1).strip()

    # Download URL (Kaggle, HuggingFace, GitHub)
    dl_m = re.search(
        r'href=["\']?(https://(?:www\.kaggle\.com|huggingface\.co|github\.com)[^\s"\'<>]+)',
        html, re.IGNORECASE
    )
    if dl_m:
        result['resource.download'] = dl_m.group(1)
        result['resource.url'] = dl_m.group(1)

    result['extras.uploadType'] = 'dataset'
    result['extras.pocName'] = 'MLCommons'
    result['extras.pocEmail'] = 'datasets@mlcommons.org'
    result['extras.publisher'] = 'MLCommons'
    result['source_format'] = 'MLCommons Web'
    result['_source_url'] = url

    return result


def _merge_with_catalog(scraped: dict, catalog_entry: dict) -> dict:
    """Merge scraped data with known catalog entry. Scraped values win if non-empty."""
    merged = dict(catalog_entry)
    for k, v in scraped.items():
        if v and str(v).strip() and k not in ('source_format', '_source_url'):
            merged[k] = v
    # Always keep these from scraped
    for k in ('source_format', '_source_url', 'title', 'notes'):
        if scraped.get(k):
            merged[k] = scraped[k]
    return merged


# ── Public API ────────────────────────────────────────────────────────────────

def get_mlcommons_catalog() -> list:
    """Return the known MLCommons dataset catalog (no network required)."""
    return [
        {
            "slug": d["slug"],
            "title": d["title"],
            "url": d["url"],
            "summary": d["notes"][:200] + "…",
            "tags": d.get("tags", []),
            "format": d.get("extras.format", ""),
            "license": d.get("extras.license", ""),
            "size": d.get("extras.size", ""),
        }
        for d in MLCOMMONS_CATALOG
    ]


def extract_mlcommons_dataset(
    slug: str,
    live_scrape: bool = True,
    progress_cb=None
) -> dict:
    """
    Extract full metadata for an MLCommons dataset.

    Args:
        slug: Dataset slug (e.g. 'peoples-speech', 'dollar-street')
        live_scrape: If True, attempt to fetch the live page for latest data.
        progress_cb: Optional callback(str) for progress messages.

    Returns:
        Flat metadata dict compatible with NDP/CKAN schema.
    """
    def log(msg):
        if progress_cb:
            progress_cb(msg)

    # Find catalog entry
    catalog_entry = next((d for d in MLCOMMONS_CATALOG if d["slug"] == slug), None)
    if not catalog_entry:
        return {"error": f"Unknown MLCommons dataset slug: {slug}"}

    url = catalog_entry["url"]

    if live_scrape:
        log(f"🌐 Fetching live page: {url}")
        html = _fetch_page(url)
        if html and len(html) > 500:
            log("📄 Parsing page content…")
            scraped = _parse_mlcommons_page(html, slug, url)
            merged = _merge_with_catalog(scraped, catalog_entry)
            merged['source_format'] = 'MLCommons (live)'
            log(f"✅ Live scrape complete — {len([v for v in merged.values() if v])} fields extracted.")
            return merged
        else:
            log("⚠️ Live page not reachable — using catalog data.")

    log("📋 Using built-in catalog data.")
    result = dict(catalog_entry)
    result['source_format'] = 'MLCommons (catalog)'
    result['_source_url'] = url
    return result


def normalize_mlcommons_to_flat(entry: dict) -> dict:
    """
    Convert an MLCommons catalog entry / scraped result to NDP flat metadata.
    Ensures all top-level extras.* and resource.* keys are present.
    """
    flat = {}

    # Core fields
    flat['title'] = entry.get('title', '')
    flat['notes'] = entry.get('notes', '')
    flat['tags'] = entry.get('tags', [])

    # Extras
    for key in ['uploadType', 'dataType', 'pocName', 'pocEmail',
                'issueDate', 'lastUpdateDate', 'license', 'publisher',
                'size', 'examples', 'format']:
        val = entry.get(f'extras.{key}', '')
        if val:
            flat[f'extras.{key}'] = val

    # Resource
    flat['resource.name'] = entry.get('resource.name', entry.get('title', ''))
    flat['resource.description'] = entry.get('resource.description', entry.get('notes', '')[:200])
    flat['resource.mimetype'] = entry.get('resource.mimetype', '')
    flat['resource.format'] = entry.get('resource.format', '')
    flat['resource.status'] = entry.get('resource.status', 'active')
    flat['resource.url'] = entry.get('resource.download') or entry.get('resource.url', '')

    flat['extras.uploadType'] = flat.get('extras.uploadType', 'dataset')
    flat['source_format'] = entry.get('source_format', 'MLCommons')
    flat['_source_url'] = entry.get('_source_url', entry.get('url', ''))

    return flat
