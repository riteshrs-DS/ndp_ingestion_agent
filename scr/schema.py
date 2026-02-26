CKAN_DATASET_SCHEMA = {
    "type": "object",
    "required": ["title", "notes", "license_id", "tags"],
    "properties": {
        "title": {"type": "string"},
        "notes": {"type": "string"},
        "license_id": {"type": "string"},
        "author": {"type": "string"},
        "maintainer": {"type": "string"},
        "owner_org": {"type": "string"},
        "url": {"type": "string"},
        "version": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
        },
        "resources": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "url", "format"],
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": "string"},
                    "format": {"type": "string"},
                    "description": {"type": "string"},
                }
            }
        }
    },
    "additionalProperties": True
}