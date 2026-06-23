"""
llm_provider.py  ·  Unified LLM interface for the NDP Ingestion Agent
───────────────────────────────────────────────────────────────────────
Thin wrapper around llm_registry.generate() that keeps the same
function signatures the rest of the app already uses (normalize,
repair, questions) but now works with ANY registered model.

Drop-in replacement for llm_ollama.py — nothing else needs to change.
"""

import json
from typing import Optional

from utils.llm_registry import (
    generate, parse_json_response,
    check_model_connectivity, check_all_ollama_models,
    DEFAULT_MODEL_KEY, MODEL_REGISTRY,
    get_registry, get_registry_by_group, has_credential,
    GROUP_META, model_label,
)

# ── Re-export for backwards compatibility with existing imports ───────────────
DEFAULT_MODEL    = "llama3"          # legacy name still used in sidebar defaults
DEFAULT_BASE_URL = "http://localhost:11434"


# ── Backwards-compat shim (used by sidebar "Check Connection" button) ─────────
def check_ollama_status(base_url: str = DEFAULT_BASE_URL) -> tuple[bool, list]:
    """Legacy shim: probe the local Ollama instance."""
    result = check_all_ollama_models({"OLLAMA_BASE_URL": base_url})
    models = result.get("models", [])
    return (bool(models), models)


# ── Prompt builders (unchanged — purely text, model-agnostic) ─────────────────

def build_normalization_prompt(raw_metadata: dict, missing_fields: list) -> str:
    raw_text      = raw_metadata.get('_raw_text', '')
    source_format = raw_metadata.get('source_format', 'unknown')
    existing      = {k: v for k, v in raw_metadata.items()
                     if not k.startswith('_') and k != 'source_format' and v}

    return f"""You are a metadata normalization expert for the National Data Platform (NDP).
Your task is to extract and normalize dataset metadata into CKAN-compatible fields.

Source format: {source_format}

Existing extracted metadata:
{json.dumps(existing, indent=2)}

Raw text content (if available):
{raw_text[:2000] if raw_text else "N/A"}

Missing required fields that need to be filled:
{json.dumps(missing_fields, indent=2)}

Please analyze the available information and provide values for the missing fields.
Return ONLY a valid JSON object with the missing field values. Use null if a field cannot be determined.

Field descriptions:
- title: Short descriptive title of the dataset
- notes: Detailed description of the dataset contents and scope
- tags: List of relevant keywords (at least 1)
- extras.uploadType: One of: dataset, service, model, collection
- extras.dataType: Data format type (e.g., tabular, timeseries, imagery, text)
- extras.pocName: Full name of the point of contact person
- extras.pocEmail: Email address of the point of contact
- extras.issueDate: Publication/creation date in YYYY-MM-DD format
- extras.lastUpdateDate: Last modification date in YYYY-MM-DD format
- resource.name: Name/title of the primary data resource
- resource.description: Brief description of the resource
- resource.mimetype: MIME type (e.g., text/csv, application/json)
- resource.format: File format (e.g., CSV, JSON, XML, NetCDF)
- resource.status: One of: active, archived, deprecated

Return ONLY valid JSON, no markdown, no explanation:"""


def build_repair_prompt(current_json: dict, validation_errors: list,
                        user_answers: dict) -> str:
    return f"""You are a metadata repair expert for the National Data Platform (NDP).

Current metadata JSON:
{json.dumps(current_json, indent=2)}

Validation errors:
{json.dumps(validation_errors, indent=2)}

User-provided values for missing fields:
{json.dumps(user_answers, indent=2)}

Please update the metadata JSON by:
1. Incorporating all user-provided values
2. Fixing any formatting issues mentioned in validation errors
3. Ensuring dates are in YYYY-MM-DD format
4. Ensuring tags is a list of objects with "name" key: [{{"name": "keyword"}}]
5. Ensuring extras is a list of {{"key": "...", "value": "..."}} objects
6. Ensuring resources is a list with at least one resource object

Return ONLY the complete, updated, valid CKAN JSON package object. No markdown, no explanation:"""


# ── Core LLM operations (called by app.py) ────────────────────────────────────

def normalize_metadata_with_llm(
    raw_metadata: dict,
    missing_fields: list,
    model_key: str = DEFAULT_MODEL_KEY,
    session_overrides: dict = None,
    # legacy kwargs kept for backwards compat
    model: str = None,
    base_url: str = None,
) -> dict:
    """
    Use the selected LLM to infer values for missing NDP metadata fields.
    Returns a dict of {field: value} for fields the model could fill.
    """
    if not missing_fields:
        return {}

    # Resolve model_key: prefer explicit model_key, fall back to legacy 'model' arg
    key = model_key
    if model and model != DEFAULT_MODEL:
        # caller is using the old (model, base_url) signature — map to a key
        key = _legacy_to_key(model, base_url)

    prompt   = build_normalization_prompt(raw_metadata, missing_fields)
    response = generate(prompt, model_key=key,
                        session_overrides=session_overrides, temperature=0.1)

    parsed = parse_json_response(response)
    if parsed:
        return {k: v for k, v in parsed.items() if v is not None and v != ""}
    return {}


def repair_metadata_with_llm(
    current_json: dict,
    validation_errors: list,
    user_answers: dict,
    model_key: str = DEFAULT_MODEL_KEY,
    session_overrides: dict = None,
    # legacy kwargs
    model: str = None,
    base_url: str = None,
) -> Optional[dict]:
    """Use the selected LLM to repair a CKAN JSON package that failed validation."""
    key = model_key
    if model and model != DEFAULT_MODEL:
        key = _legacy_to_key(model, base_url)

    prompt   = build_repair_prompt(current_json, validation_errors, user_answers)
    response = generate(prompt, model_key=key,
                        session_overrides=session_overrides,
                        temperature=0.1, max_tokens=3000)
    return parse_json_response(response)


def generate_questions_for_missing_fields(missing_fields: list) -> dict:
    """Return human-readable questions for each missing NDP field (no LLM needed)."""
    questions = {
        "title":                "What is the title of your dataset?",
        "notes":                "Please provide a detailed description (contents, scope, purpose):",
        "tags":                 "Enter keywords/tags (comma-separated, at least 1 required):",
        "extras.uploadType":    "What type of resource? (dataset / service / model / collection)",
        "extras.dataType":      "What is the data type? (tabular, timeseries, imagery, text, geospatial)",
        "extras.pocName":       "Who is the point of contact? (Full name)",
        "extras.pocEmail":      "Contact email for the point of contact?",
        "extras.issueDate":     "When was this dataset first published? (YYYY-MM-DD)",
        "extras.lastUpdateDate":"When was it last updated? (YYYY-MM-DD)",
        "resource.name":        "Name/title of the primary data resource?",
        "resource.description": "Briefly describe the resource (file/API/link):",
        "resource.mimetype":    "MIME type of the resource? (e.g., text/csv, application/json)",
        "resource.format":      "File format? (e.g., CSV, JSON, XML, NetCDF, ZIP)",
        "resource.status":      "Status of this dataset? (active / archived / deprecated)",
        "resource.url":         "URL to access or download this resource?",
    }
    return {f: questions.get(f, f"Please provide a value for: {f}")
            for f in missing_fields}


# ── Legacy key resolver ────────────────────────────────────────────────────────

def _legacy_to_key(model_name: str, base_url: str = None) -> str:
    """
    Map an old-style (model_name, base_url) pair to a registry key.
    Falls back to ollama/llama3 if no match found.
    """
    for entry in MODEL_REGISTRY:
        if entry["model"] == model_name:
            if base_url and entry["base_url"] in (base_url, base_url.rstrip("/v1").rstrip("/")):
                return entry["key"]
            if not base_url:
                return entry["key"]
    return DEFAULT_MODEL_KEY
