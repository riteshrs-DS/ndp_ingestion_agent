"""
NDP Validation: Integrates with the NDP validation endpoint.
Handles preflight checks and full validation via API.
"""
import json
import re
import requests
from typing import Optional
from utils.schema import PUBLIC_REQUIRED_FIELDS, get_missing_required_fields


NDP_VALIDATE_URL = "https://ndp-test.sdsc.edu/catalog2/ndp/package_validate"


def build_ckan_package(flat_metadata: dict, is_private: bool = False, owner_org: str = "") -> dict:
    """Convert flat metadata dict into a proper CKAN package JSON."""
    import re
    from datetime import datetime

    def make_slug(title: str) -> str:
        slug = title.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'\s+', '-', slug.strip())
        slug = re.sub(r'-+', '-', slug)
        return slug[:100] or "unnamed-dataset"

    title = flat_metadata.get('title', 'Unnamed Dataset')
    package = {
        "name": make_slug(title),
        "title": title,
        "notes": flat_metadata.get('notes', ''),
        "private": is_private,
        "owner_org": owner_org or "ndp",
    }

    # Tags
    tags_raw = flat_metadata.get('tags', [])
    if isinstance(tags_raw, str):
        tags_raw = [t.strip() for t in re.split(r'[,;]+', tags_raw) if t.strip()]
    package["tags"] = [{"name": re.sub(r'[^a-zA-Z0-9\s\-_.]', '', t)[:100]} for t in tags_raw if t]

    # Extras
    extra_keys = [
        'uploadType', 'dataType', 'pocName', 'pocEmail',
        'issueDate', 'lastUpdateDate', 'temporal'
    ]
    extras = []
    for key in extra_keys:
        val = flat_metadata.get(f'extras.{key}', '')
        if val:
            extras.append({"key": key, "value": str(val)})
    package["extras"] = extras

    # Resources
    resource = {
        "name": flat_metadata.get('resource.name', title),
        "description": flat_metadata.get('resource.description', ''),
        "mimetype": flat_metadata.get('resource.mimetype', ''),
        "format": flat_metadata.get('resource.format', ''),
        "url": flat_metadata.get('resource.url', 'https://ndp-test.sdsc.edu/catalog'),
        "status": flat_metadata.get('resource.status', 'active'),
    }
    # Remove empty resource fields
    resource = {k: v for k, v in resource.items() if v}
    if 'url' not in resource:
        resource['url'] = 'https://ndp-test.sdsc.edu/catalog'
    if 'name' not in resource:
        resource['name'] = title

    package["resources"] = [resource]
    return package


def preflight_check(flat_metadata: dict, is_private: bool = False) -> dict:
    """Run local preflight validation before calling the API."""
    missing = get_missing_required_fields(
        build_ckan_package(flat_metadata, is_private=is_private),
        is_private=is_private
    )

    # Additional checks
    warnings = []
    errors = []

    title = flat_metadata.get('title', '')
    if title and len(title) < 5:
        warnings.append("Title seems very short (< 5 characters)")

    notes = flat_metadata.get('notes', '')
    if notes and len(notes) < 20:
        warnings.append("Description seems very short (< 20 characters)")

    email = flat_metadata.get('extras.pocEmail', '')
    if email and '@' not in email:
        errors.append("Point of contact email does not appear to be valid")

    date_val = flat_metadata.get('extras.issueDate', '')
    if date_val:
        import re
        if not re.match(r'\d{4}-\d{2}-\d{2}', date_val):
            warnings.append(f"Issue date '{date_val}' should be in YYYY-MM-DD format")

    tags = flat_metadata.get('tags', [])
    if isinstance(tags, list) and len(tags) == 0 and not is_private:
        errors.append("At least 1 tag is required for public datasets")

    return {
        "missing_fields": missing,
        "errors": errors,
        "warnings": warnings,
        "passed": len(missing) == 0 and len(errors) == 0
    }


def validate_with_ndp_api(ckan_package: dict) -> dict:
    """Call the NDP validation API endpoint."""
    try:
        resp = requests.post(
            NDP_VALIDATE_URL,
            json=ckan_package,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        result = {
            "status_code": resp.status_code,
            "response_text": resp.text,
        }
        try:
            result["response_json"] = resp.json()
        except Exception:
            result["response_json"] = None

        result["passed"] = resp.status_code in (200, 201)
        return result
    except requests.exceptions.ConnectionError:
        return {
            "status_code": None,
            "passed": False,
            "error": "Could not connect to NDP validation endpoint. Check network access.",
            "response_text": "",
            "response_json": None
        }
    except Exception as e:
        return {
            "status_code": None,
            "passed": False,
            "error": str(e),
            "response_text": "",
            "response_json": None
        }


def parse_validation_errors(api_result: dict) -> list:
    """Extract error messages from API validation result."""
    errors = []

    if api_result.get("error"):
        errors.append(api_result["error"])
        return errors

    resp_json = api_result.get("response_json", {})
    resp_text = api_result.get("response_text", "")

    if resp_json:
        # Try common error formats
        if isinstance(resp_json, dict):
            if "error" in resp_json:
                err = resp_json["error"]
                if isinstance(err, dict):
                    for k, v in err.items():
                        if k != "__type":
                            errors.append(f"{k}: {v}")
                elif isinstance(err, str):
                    errors.append(err)
            if "errors" in resp_json:
                for e in resp_json["errors"]:
                    errors.append(str(e))
            if "message" in resp_json:
                errors.append(resp_json["message"])

    if not errors and resp_text and not api_result.get("passed"):
        errors.append(f"Validation failed (HTTP {api_result.get('status_code')}): {resp_text[:500]}")

    return errors


def format_validation_report(preflight: dict, api_result: Optional[dict] = None) -> str:
    """Format a human-readable validation report."""
    lines = []

    lines.append("## Preflight Check")
    if preflight["passed"]:
        lines.append("✅ Preflight check passed: no missing required fields detected.")
    else:
        lines.append("❌ Preflight check FAILED:")
        if preflight["missing_fields"]:
            lines.append("**Missing required fields:**")
            for f in preflight["missing_fields"]:
                lines.append(f"  - {f}")
        if preflight["errors"]:
            lines.append("**Errors:**")
            for e in preflight["errors"]:
                lines.append(f"  - {e}")
        if preflight["warnings"]:
            lines.append("**Warnings:**")
            for w in preflight["warnings"]:
                lines.append(f"  - ⚠️ {w}")

    if api_result:
        lines.append("\n## NDP API Validation")
        if api_result.get("passed"):
            lines.append("✅ NDP validation PASSED!")
        else:
            lines.append("❌ NDP validation FAILED")
            errors = parse_validation_errors(api_result)
            for e in errors:
                lines.append(f"  - {e}")

    return "\n".join(lines)
