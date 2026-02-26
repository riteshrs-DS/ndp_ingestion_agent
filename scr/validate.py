from jsonschema import validate
from jsonschema.exceptions import ValidationError
from typing import Tuple, Optional, Any, Dict

def validate_json(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e)