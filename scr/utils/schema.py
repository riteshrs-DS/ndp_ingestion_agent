"""
NDP/CKAN Metadata Schema Definitions
Based on the required fields from the NDP validation endpoint
and the spreadsheet screenshot provided.
"""

# Required fields for PUBLIC datasets
PUBLIC_REQUIRED_FIELDS = {
    "title": {
        "description": "The title of the dataset, resource, or dataset version.",
        "ckan_field": "title",
        "data_type": "text",
        "section": "General"
    },
    "notes": {
        "description": "A detailed explanation of the dataset, its contents, and its scope.",
        "ckan_field": "notes",
        "data_type": "text",
        "section": "General"
    },
    "tags": {
        "description": "A set of terms or phrases that describe the dataset (at least 1 tag required for public datasets).",
        "ckan_field": "tags",
        "data_type": "list",
        "section": "General"
    },
    "extras.uploadType": {
        "description": "Defines the nature of the entity or resource (e.g., dataset, service, model).",
        "ckan_field": "extras:uploadType",
        "data_type": "text",
        "section": "General"
    },
    "extras.dataType": {
        "description": "Describes the format of the data (e.g., zip, csv, api).",
        "ckan_field": "extras:dataType",
        "data_type": "text",
        "section": "General"
    },
    "extras.pocName": {
        "description": "The name of the point of contact for the dataset.",
        "ckan_field": "extras:pocName",
        "data_type": "text",
        "section": "Contributors and Contact"
    },
    "extras.pocEmail": {
        "description": "The contact email for the point of contact.",
        "ckan_field": "extras:pocEmail",
        "data_type": "text",
        "section": "Contributors and Contact"
    },
    "extras.issueDate": {
        "description": "The date the dataset was first made publicly available or published.",
        "ckan_field": "extras:issueDate",
        "data_type": "text",
        "section": "General Metadata"
    },
    "extras.lastUpdateDate": {
        "description": "The date when the dataset was last updated or modified.",
        "ckan_field": "extras:lastUpdateDate",
        "data_type": "text",
        "section": "General Metadata"
    },
    "resource.name": {
        "description": "The title of a specific resource within the dataset.",
        "ckan_field": "resource:name",
        "data_type": "text",
        "section": "Resource"
    },
    "resource.description": {
        "description": "A brief description of the resource being described.",
        "ckan_field": "resource:description",
        "data_type": "text",
        "section": "Resource"
    },
    "resource.mimetype": {
        "description": "The MIME type of the resource (e.g., application/json, text/csv).",
        "ckan_field": "resource:mimetype",
        "data_type": "text",
        "section": "Resource"
    },
    "resource.format": {
        "description": "The file format in which the dataset is stored (e.g., CSV, JSON, XML, NetCDF).",
        "ckan_field": "N/A",
        "data_type": "text",
        "section": "Resource"
    },
    "resource.status": {
        "description": "Indicates whether the dataset is active, archived, or deprecated.",
        "ckan_field": "resource:status",
        "data_type": "text",
        "section": "Resource"
    },
}

# Required fields for PRIVATE datasets (subset)
PRIVATE_REQUIRED_FIELDS = {
    "title", "notes", "extras.uploadType",
    "extras.issueDate", "extras.lastUpdateDate",
    "resource.name", "resource.description"
}

# Optional fields for public
PUBLIC_OPTIONAL_FIELDS = {
    "tags", "extras.dataType", "extras.pocName", "extras.pocEmail",
    "resource.mimetype", "resource.format", "resource.status"
}

# Valid values for certain fields
VALID_UPLOAD_TYPES = ["dataset", "service", "model", "collection"]
VALID_RESOURCE_STATUSES = ["active", "archived", "deprecated"]
VALID_MIME_TYPES = [
    "text/csv", "application/json", "text/xml", "application/xml",
    "application/pdf", "text/plain", "application/zip",
    "application/x-netcdf", "application/octet-stream"
]

def get_empty_ckan_package():
    """Return a blank CKAN-compatible package template."""
    return {
        "name": "",
        "title": "",
        "notes": "",
        "tags": [],
        "owner_org": "",
        "private": False,
        "extras": [
            {"key": "uploadType", "value": "dataset"},
            {"key": "dataType", "value": ""},
            {"key": "pocName", "value": ""},
            {"key": "pocEmail", "value": ""},
            {"key": "issueDate", "value": ""},
            {"key": "lastUpdateDate", "value": ""},
        ],
        "resources": [
            {
                "name": "",
                "description": "",
                "mimetype": "",
                "format": "",
                "status": "active",
                "url": ""
            }
        ]
    }

def flatten_ckan_package(package: dict) -> dict:
    """Flatten a CKAN package into a simple key-value dict for display."""
    flat = {}
    flat["title"] = package.get("title", "")
    flat["notes"] = package.get("notes", "")
    flat["tags"] = [t.get("name", t) if isinstance(t, dict) else t for t in package.get("tags", [])]

    for extra in package.get("extras", []):
        flat[f"extras.{extra['key']}"] = extra.get("value", "")

    resources = package.get("resources", [])
    if resources:
        r = resources[0]
        flat["resource.name"] = r.get("name", "")
        flat["resource.description"] = r.get("description", "")
        flat["resource.mimetype"] = r.get("mimetype", "")
        flat["resource.format"] = r.get("format", "")
        flat["resource.status"] = r.get("status", "active")
        flat["resource.url"] = r.get("url", "")

    return flat

def get_missing_required_fields(package: dict, is_private: bool = False) -> list:
    """Return list of missing required fields from a CKAN package."""
    flat = flatten_ckan_package(package)
    missing = []
    required = PRIVATE_REQUIRED_FIELDS if is_private else set(PUBLIC_REQUIRED_FIELDS.keys())

    for field in required:
        val = flat.get(field)
        if not val or (isinstance(val, list) and len(val) == 0) or (isinstance(val, str) and val.strip() == ""):
            missing.append(field)

    return missing
