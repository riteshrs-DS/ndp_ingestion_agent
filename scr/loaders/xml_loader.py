"""
XML Loader: Handles ISO/XML and EML metadata formats from NDP CKAN catalog.
Includes deep extraction: fetches the actual source XML/TXT behind each CKAN package.
"""
import re
import requests
from xml.etree import ElementTree as ET
from typing import Optional, Callable
import json


# XML namespace maps
ISO_NS = {
    'gmd': 'http://www.isotc211.org/2005/gmd',
    'gco': 'http://www.isotc211.org/2005/gco',
    'gml': 'http://www.opengis.net/gml',
    'gmx': 'http://www.isotc211.org/2005/gmx',
    'xlink': 'http://www.w3.org/1999/xlink',
    'gmi': 'http://www.isotc211.org/2005/gmi',
    'srv': 'http://www.isotc211.org/2005/srv',
}

EML_NS = {
    'eml': 'eml://ecoinformatics.org/eml-2.1.1',
    'stmml': 'http://www.xml-cml.org/schema/stmml-1.1',
}


def safe_find_text(element, xpath: str, ns: dict, default="") -> str:
    """Safely extract text from an XML element."""
    try:
        el = element.find(xpath, ns)
        if el is not None and el.text:
            return el.text.strip()
    except Exception:
        pass
    return default


def safe_findall(element, xpath: str, ns: dict) -> list:
    """Safely find all matching elements."""
    try:
        return element.findall(xpath, ns)
    except Exception:
        return []


def parse_iso_xml(xml_content: str) -> dict:
    """Parse ISO 19139 XML metadata into normalized dict."""
    try:
        root = ET.fromstring(xml_content.encode('utf-8') if isinstance(xml_content, str) else xml_content)
    except ET.ParseError as e:
        return {"error": str(e)}

    ns = ISO_NS
    result = {}

    # Title
    title_paths = [
        './/gmd:identificationInfo//gmd:citation//gmd:title/gco:CharacterString',
        './/gmd:title/gco:CharacterString',
    ]
    for path in title_paths:
        val = safe_find_text(root, path, ns)
        if val:
            result['title'] = val
            break

    # Abstract/Notes
    abstract_paths = [
        './/gmd:identificationInfo//gmd:abstract/gco:CharacterString',
        './/gmd:abstract/gco:CharacterString',
    ]
    for path in abstract_paths:
        val = safe_find_text(root, path, ns)
        if val:
            result['notes'] = val
            break

    # Keywords
    keywords = []
    kw_elements = safe_findall(root, './/gmd:keyword/gco:CharacterString', ns)
    for kw in kw_elements:
        if kw.text and kw.text.strip():
            keywords.append(kw.text.strip())
    if keywords:
        result['tags'] = list(set(keywords))

    # Date
    date_paths = [
        './/gmd:CI_Date/gmd:date/gco:DateTime',
        './/gmd:CI_Date/gmd:date/gco:Date',
        './/gmd:dateStamp/gco:DateTime',
        './/gmd:dateStamp/gco:Date',
    ]
    for path in date_paths:
        val = safe_find_text(root, path, ns)
        if val:
            result['extras.issueDate'] = val[:10] if len(val) > 10 else val
            result['extras.lastUpdateDate'] = val[:10] if len(val) > 10 else val
            break

    # Point of Contact
    poc_paths = {
        'name': [
            './/gmd:contact//gmd:individualName/gco:CharacterString',
            './/gmd:identificationInfo//gmd:pointOfContact//gmd:individualName/gco:CharacterString',
        ],
        'email': [
            './/gmd:contact//gmd:electronicMailAddress/gco:CharacterString',
            './/gmd:identificationInfo//gmd:pointOfContact//gmd:electronicMailAddress/gco:CharacterString',
        ],
        'org': [
            './/gmd:contact//gmd:organisationName/gco:CharacterString',
            './/gmd:identificationInfo//gmd:pointOfContact//gmd:organisationName/gco:CharacterString',
        ]
    }
    for key, paths in poc_paths.items():
        for path in paths:
            val = safe_find_text(root, path, ns)
            if val:
                if key == 'name':
                    result['extras.pocName'] = val
                elif key == 'email':
                    result['extras.pocEmail'] = val
                elif key == 'org':
                    result['extras.pocOrg'] = val
                break

    # Format / Resource
    fmt_paths = [
        './/gmd:distributionFormat//gmd:name/gco:CharacterString',
        './/gmd:MD_Format/gmd:name/gco:CharacterString',
    ]
    for path in fmt_paths:
        val = safe_find_text(root, path, ns)
        if val:
            result['resource.format'] = val
            break

    # Resource URL
    url_paths = [
        './/gmd:onLine//gmd:linkage/gmd:URL',
        './/gmd:CI_OnlineResource/gmd:linkage/gmd:URL',
    ]
    for path in url_paths:
        val = safe_find_text(root, path, ns)
        if val:
            result['resource.url'] = val
            break

    result['extras.uploadType'] = 'dataset'
    result['source_format'] = 'ISO/XML'
    return result


def parse_eml_xml(xml_content: str) -> dict:
    """Parse EML (Ecological Metadata Language) XML into normalized dict."""
    try:
        root = ET.fromstring(xml_content.encode('utf-8') if isinstance(xml_content, str) else xml_content)
    except ET.ParseError as e:
        return {"error": str(e)}

    result = {}

    def find_text(el, path, default=""):
        found = el.find(path)
        return found.text.strip() if found is not None and found.text else default

    # Title
    title = find_text(root, './/dataset/title')
    if not title:
        title = find_text(root, './/title')
    if title:
        result['title'] = title

    # Abstract
    abstract = find_text(root, './/dataset/abstract/para')
    if not abstract:
        abstract = find_text(root, './/abstract/para')
    if not abstract:
        abstract = find_text(root, './/abstract')
    if abstract:
        result['notes'] = abstract

    # Keywords
    keywords = []
    for kw in root.findall('.//keywordSet/keyword'):
        if kw.text:
            keywords.append(kw.text.strip())
    if keywords:
        result['tags'] = list(set(keywords))

    # Creator / Contact
    creator_name = find_text(root, './/creator/individualName/givenName')
    creator_surname = find_text(root, './/creator/individualName/surName')
    if creator_name or creator_surname:
        result['extras.pocName'] = f"{creator_name} {creator_surname}".strip()

    creator_email = find_text(root, './/creator/electronicMailAddress')
    if not creator_email:
        creator_email = find_text(root, './/contact/electronicMailAddress')
    if creator_email:
        result['extras.pocEmail'] = creator_email

    # Dates
    pub_date = find_text(root, './/dataset/pubDate')
    if pub_date:
        result['extras.issueDate'] = pub_date
        result['extras.lastUpdateDate'] = pub_date

    # Data format
    data_format = find_text(root, './/dataTable/dataFormat/textFormat/simpleDelimited/fieldDelimiter')
    if data_format:
        result['resource.format'] = 'CSV'
        result['resource.mimetype'] = 'text/csv'

    result['extras.uploadType'] = 'dataset'
    result['source_format'] = 'EML'
    return result


def fetch_ckan_packages(base_url: str, org: str, max_rows: int = 5) -> list:
    """Fetch dataset packages from a CKAN organization."""
    try:
        url = f"{base_url}/api/3/action/package_search"
        params = {"fq": f"organization:{org}", "rows": max_rows, "start": 0}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return data["result"]["results"]
    except Exception as e:
        return [{"error": str(e)}]
    return []



def normalize_from_ckan_api(package: dict) -> dict:
    """
    Normalize a CKAN API package result into our flat format.
    Reads top-level fields, extras, and resources.
    """
    result = {}
    result['title'] = package.get('title', '')
    result['notes'] = package.get('notes', '')

    tags = package.get('tags', [])
    result['tags'] = [t.get('name', '') for t in tags if isinstance(t, dict)]

    extras = package.get('extras', [])
    for extra in extras:
        key = extra.get('key', '')
        val = extra.get('value', '')
        if key and val:
            result[f'extras.{key}'] = val

    resources = package.get('resources', [])
    if resources:
        r = resources[0]
        result['resource.name'] = r.get('name', '')
        result['resource.description'] = r.get('description', '')
        result['resource.mimetype'] = r.get('mimetype', '')
        result['resource.format'] = r.get('format', '')
        result['resource.url'] = r.get('url', '')
        # status may live in extras dict or as direct key
        if isinstance(r.get('extras'), dict):
            result['resource.status'] = r['extras'].get('status', 'active')
        else:
            result['resource.status'] = r.get('status', 'active')

    result['extras.uploadType'] = result.get('extras.uploadType', 'dataset')
    result['source_format'] = 'CKAN API'
    return result


# ─── Deep extraction from source URLs ────────────────────────────────────────

def _detect_and_parse_xml(content: str) -> Optional[dict]:
    """Auto-detect ISO vs EML and parse accordingly."""
    c = content.strip()
    if not c.startswith('<'):
        return None
    low = c[:500].lower()
    if 'eml://ecoinformatics' in low or '<eml:eml' in low or 'ecoinformatics' in low:
        result = parse_eml_xml(content)
    else:
        result = parse_iso_xml(content)
    return result if not result.get('error') else None


def _fetch_url_content(url: str, timeout: int = 20) -> Optional[str]:
    """Fetch raw content from a URL."""
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={'User-Agent': 'NDP-Ingestion-Agent/1.0'})
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def _try_iso_endpoint(base_url: str, pkg_name: str) -> Optional[dict]:
    """Try fetching the ISO XML endpoint for a CKAN package."""
    candidates = [
        f"{base_url}/dataset/{pkg_name}/iso",
        f"{base_url}/dataset/{pkg_name}/export/iso19139.xml",
        f"{base_url}/dataset/{pkg_name}.xml",
    ]
    for url in candidates:
        content = _fetch_url_content(url)
        if content and content.strip().startswith('<'):
            result = parse_iso_xml(content)
            if result and not result.get('error') and result.get('title'):
                result['_deep_source_url'] = url
                result['_deep_source_type'] = 'ISO XML endpoint'
                return result
    return None


def _try_resource_urls(resources: list) -> Optional[dict]:
    """
    Try to fetch and parse XML/TXT/EML metadata from resource URLs.
    Returns the first successfully parsed result.
    """
    for r in resources:
        url = r.get('url', '')
        fmt = (r.get('format') or '').upper()
        mimetype = (r.get('mimetype') or '').lower()
        name = (r.get('name') or '').lower()

        # Skip obviously non-metadata resources (data files, pelican://, s3://)
        if not url or not url.startswith('http'):
            continue
        if any(x in url for x in ['pelican://', 's3://', 'ftp://']):
            continue
        if fmt in ('CSV', 'ZIP', 'PDF', 'NETCDF', 'NC', 'HDF5', 'TIFF', 'PNG', 'JPG'):
            continue

        # Prefer XML/EML/TXT resources
        is_xml = fmt in ('XML', 'EML', 'ISO') or 'xml' in mimetype or url.endswith('.xml')
        is_txt = fmt == 'TXT' or 'text/plain' in mimetype or url.endswith('.txt')
        is_meta = 'metadata' in name or 'meta' in name or is_xml or is_txt

        if not is_meta:
            continue

        content = _fetch_url_content(url)
        if not content:
            continue

        content_stripped = content.strip()
        if content_stripped.startswith('<'):
            parsed = _detect_and_parse_xml(content_stripped)
            if parsed and parsed.get('title'):
                parsed['_deep_source_url'] = url
                parsed['_deep_source_type'] = f'Resource XML ({fmt or "auto"})'
                return parsed
        elif len(content_stripped) > 30:
            from loaders.text_loader import parse_txt_metadata, clean_text
            parsed = parse_txt_metadata(clean_text(content_stripped))
            if parsed.get('title') or parsed.get('notes'):
                parsed['_deep_source_url'] = url
                parsed['_deep_source_type'] = f'Resource TXT ({fmt or "text"})'
                return parsed
    return None


def _try_package_show(base_url: str, pkg_name: str) -> Optional[dict]:
    """Fetch full package detail via CKAN package_show API."""
    try:
        url = f"{base_url}/api/3/action/package_show"
        resp = requests.get(url, params={"id": pkg_name}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                return data["result"]
    except Exception:
        pass
    return None


def deep_extract_from_ckan_package(
    package: dict,
    base_url: str = "https://ndp-test.sdsc.edu/catalog",
    org: str = "",
    progress_cb: Optional[Callable[[str], None]] = None
) -> dict:
    """
    Full deep extraction pipeline for a CKAN package:

    1. Fetch enriched package_show detail (more extras than package_search)
    2. Try ISO XML endpoint for the package
    3. Try resource URLs for XML/EML/TXT content
    4. Fall back to top-level CKAN API fields
    5. Merge all layers, preferring deeper sources

    Returns a comprehensive flat metadata dict.
    """
    def log(msg):
        if progress_cb:
            progress_cb(msg)

    pkg_name = package.get('name', '')
    pkg_id   = package.get('id', pkg_name)

    # ── Layer 0: Fetch full package_show (richer than search result) ──────────
    log(f"🔍 Fetching full package detail for `{pkg_name}`…")
    full_pkg = _try_package_show(base_url, pkg_name) or package

    # Start with the basic CKAN-API normalisation as our baseline
    base_result = normalize_from_ckan_api(full_pkg)
    base_result['source_format'] = 'CKAN API'

    # ── Layer 1: ISO XML endpoint ─────────────────────────────────────────────
    log("📡 Trying ISO XML endpoint…")
    iso_result = _try_iso_endpoint(base_url, pkg_name)
    if iso_result and iso_result.get('title'):
        log(f"✅ ISO XML found at: {iso_result.get('_deep_source_url')}")
        merged = _merge_metadata(base_result, iso_result)
        merged['source_format'] = 'CKAN+ISO/XML'
        return merged

    # ── Layer 2: Resource URLs ────────────────────────────────────────────────
    resources = full_pkg.get('resources', [])
    if resources:
        log(f"🔗 Scanning {len(resources)} resource URL(s) for metadata content…")
        res_result = _try_resource_urls(resources)
        if res_result and (res_result.get('title') or res_result.get('notes')):
            log(f"✅ Metadata extracted from resource: {res_result.get('_deep_source_url')}")
            merged = _merge_metadata(base_result, res_result)
            merged['source_format'] = f"CKAN+{res_result.get('source_format', 'Resource')}"
            return merged

    # ── Layer 3: Plain-text notes as TXT parse ────────────────────────────────
    notes = full_pkg.get('notes', '')
    if notes and len(notes) > 50:
        log("📝 Enriching from package notes text…")
        from loaders.text_loader import parse_txt_metadata, clean_text
        txt_result = parse_txt_metadata(clean_text(notes))
        merged = _merge_metadata(base_result, txt_result)
        merged['source_format'] = 'CKAN+Notes'
        # Always keep the CKAN title/notes (they are canonical)
        merged['title'] = base_result.get('title') or merged.get('title', '')
        merged['notes'] = base_result.get('notes') or merged.get('notes', '')
        log("✅ Notes enrichment complete.")
        return merged

    log("ℹ️ Using CKAN API fields only (no deeper source found).")
    return base_result


def _merge_metadata(base: dict, deep: dict) -> dict:
    """
    Merge two flat metadata dicts.
    - Deep source wins for fields it provides non-empty values for.
    - Base fills in anything deep didn't find.
    - Internal/private keys (starting with _) are kept from both.
    """
    merged = dict(base)
    for k, v in deep.items():
        if k.startswith('_'):
            merged[k] = v
            continue
        # Deep wins if it has a real value and base does not, or both do (prefer deep)
        if v and (not merged.get(k) or merged.get(k) in ('', [], {})):
            merged[k] = v
    return merged


def fetch_ckan_packages(base_url: str, org: str, max_rows: int = 5) -> list:
    """Fetch dataset packages from a CKAN organization."""
    try:
        url = f"{base_url}/api/3/action/package_search"
        params = {"fq": f"organization:{org}", "rows": max_rows, "start": 0}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return data["result"]["results"]
    except Exception as e:
        return [{"error": str(e)}]
    return []
