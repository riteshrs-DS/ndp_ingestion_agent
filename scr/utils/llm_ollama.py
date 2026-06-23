"""
Ollama LLM Integration for metadata normalization and enrichment.
Uses llama3 (or configurable model) running locally via Ollama.
"""
import json
import re
import requests
from typing import Optional


DEFAULT_MODEL = "llama3"
DEFAULT_BASE_URL = "http://localhost:11434"


def check_ollama_status(base_url: str = DEFAULT_BASE_URL) -> tuple[bool, str]:
    """Check if Ollama is running and return available models."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m['name'] for m in data.get('models', [])]
            return True, models
        return False, []
    except Exception as e:
        return False, []


def ollama_generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    stream: bool = False
) -> Optional[str]:
    """Call Ollama generate endpoint."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        resp = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=120
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        return None
    except Exception as e:
        return None


def build_normalization_prompt(raw_metadata: dict, missing_fields: list) -> str:
    """Build the LLM prompt for metadata normalization."""
    raw_text = raw_metadata.get('_raw_text', '')
    source_format = raw_metadata.get('source_format', 'unknown')

    # What we already have
    existing = {k: v for k, v in raw_metadata.items()
                if not k.startswith('_') and k != 'source_format' and v}

    prompt = f"""You are a metadata normalization expert for the National Data Platform (NDP).
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

    return prompt


def normalize_metadata_with_llm(
    raw_metadata: dict,
    missing_fields: list,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL
) -> dict:
    """Use LLM to fill in missing metadata fields."""
    if not missing_fields:
        return {}

    prompt = build_normalization_prompt(raw_metadata, missing_fields)
    response = ollama_generate(prompt, model=model, base_url=base_url, temperature=0.1)

    if not response:
        return {}

    # Clean and parse JSON
    try:
        # Remove markdown code blocks if present
        clean = re.sub(r'```(?:json)?', '', response).strip()
        clean = re.sub(r'```', '', clean).strip()

        # Find JSON object
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            # Remove null values
            return {k: v for k, v in parsed.items() if v is not None and v != ""}
    except (json.JSONDecodeError, AttributeError):
        pass

    return {}


def build_repair_prompt(current_json: dict, validation_errors: list, user_answers: dict) -> str:
    """Build prompt to repair JSON based on validation errors and user input."""
    prompt = f"""You are a metadata repair expert for the National Data Platform (NDP).

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
    return prompt


def repair_metadata_with_llm(
    current_json: dict,
    validation_errors: list,
    user_answers: dict,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL
) -> Optional[dict]:
    """Use LLM to repair metadata JSON."""
    prompt = build_repair_prompt(current_json, validation_errors, user_answers)
    response = ollama_generate(prompt, model=model, base_url=base_url, temperature=0.1, max_tokens=3000)

    if not response:
        return None

    try:
        clean = re.sub(r'```(?:json)?', '', response).strip()
        clean = re.sub(r'```', '', clean).strip()
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    return None


def generate_questions_for_missing_fields(missing_fields: list) -> dict:
    """Generate human-readable questions for missing fields."""
    questions = {
        "title": "What is the title of your dataset?",
        "notes": "Please provide a detailed description of the dataset (contents, scope, purpose):",
        "tags": "Enter keywords/tags for this dataset (comma-separated, at least 1 required):",
        "extras.uploadType": "What type of resource is this? (dataset / service / model / collection)",
        "extras.dataType": "What is the data type? (e.g., tabular, timeseries, imagery, text, geospatial)",
        "extras.pocName": "Who is the point of contact for this dataset? (Full name)",
        "extras.pocEmail": "What is the contact email for the point of contact?",
        "extras.issueDate": "When was this dataset first published/created? (YYYY-MM-DD format)",
        "extras.lastUpdateDate": "When was this dataset last updated? (YYYY-MM-DD format)",
        "resource.name": "What is the name/title of the primary data resource?",
        "resource.description": "Briefly describe the primary resource (file/API/link):",
        "resource.mimetype": "What is the MIME type of the resource? (e.g., text/csv, application/json)",
        "resource.format": "What is the file format? (e.g., CSV, JSON, XML, NetCDF, ZIP)",
        "resource.status": "What is the status of this dataset? (active / archived / deprecated)",
        "resource.url": "What is the URL to access or download this resource?",
    }
    return {f: questions.get(f, f"Please provide a value for: {f}") for f in missing_fields}
