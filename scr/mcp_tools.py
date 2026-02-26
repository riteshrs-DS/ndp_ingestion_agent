import os, json
from typing import Dict, Any, Tuple, Optional

from .llm_ollama import OllamaClient
from .schema import CKAN_DATASET_SCHEMA
from .validate import validate_json
from .utils import extract_first_json_object

from .loaders.pdf_loader import PdfLoader
from .loaders.text_loader import TextLoader
from .loaders.xml_loader import XmlLoader

def tool_save_bytes(uploaded_bytes: bytes, out_dir: str, filename: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        f.write(uploaded_bytes)
    return path

def tool_extract_facts(path: str, input_type: str, max_pdf_pages: int = 3) -> Dict[str, Any]:
    input_type = input_type.lower().strip()

    if input_type == "pdf":
        loader = PdfLoader(max_pages=max_pdf_pages)
    elif input_type in ("txt", "text"):
        loader = TextLoader()
    elif input_type in ("xml", "eml"):
        loader = XmlLoader()
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

    return loader.load(path)

def tool_generate_ckan_json(facts: Dict[str, Any], llm: OllamaClient) -> Dict[str, Any]:
    system = (
        "You are a metadata normalization agent. "
        "Return ONLY valid JSON (no markdown). "
        "Create a CKAN dataset JSON from the provided facts."
    )

    prompt = f"""
Convert the following facts into CKAN dataset JSON.

Required:
- title (string)
- notes (string)
- license_id (string; use "unknown" if unclear)
- tags (array of objects: {{ "name": "keyword" }})
Optional:
- author, maintainer, owner_org, url, version, resources[]

FACTS:
{json.dumps(facts, indent=2)[:180000]}
"""

    raw = llm.generate(prompt=prompt, system=system)
    data = extract_first_json_object(raw)

    data.setdefault("title", "")
    data.setdefault("notes", "")
    data.setdefault("license_id", "unknown")
    data.setdefault("tags", [])
    if not isinstance(data["tags"], list):
        data["tags"] = []

    return data

def tool_validate_metadata(metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    return validate_json(metadata, CKAN_DATASET_SCHEMA)

def tool_save_json(metadata: Dict[str, Any], out_dir: str, base_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return out_path
