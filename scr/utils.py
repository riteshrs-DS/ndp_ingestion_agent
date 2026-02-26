import json
from typing import Any, Dict

def extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Finds the first {...} JSON object in a string and parses it.
    Raises ValueError if not found / not valid JSON.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")

    candidate = text[start:end+1].strip()
    return json.loads(candidate)