"""
Text Loader: Handles plain-text metadata (TXT format) from NDP CKAN catalog.
"""
import re
from typing import Optional


# Heuristic patterns for extracting fields from free text
TITLE_PATTERNS = [
    r'(?:title|dataset\s*title|name)\s*[:=]\s*(.+)',
    r'^(?:Title|TITLE)\s*:\s*(.+)',
]

ABSTRACT_PATTERNS = [
    r'(?:abstract|description|summary|notes)\s*[:=]\s*(.+?)(?=\n[A-Z]|\n\n|$)',
    r'(?:Abstract|Description|Summary)\s*:\s*(.+)',
]

KEYWORD_PATTERNS = [
    r'(?:keywords?|tags?|subjects?)\s*[:=]\s*(.+)',
    r'Keywords?\s*:\s*(.+)',
]

DATE_PATTERNS = [
    r'(?:date|publication\s*date|created|issued|published)\s*[:=]\s*(\d{4}[-/]\d{2}[-/]\d{2}|\d{4})',
    r'(?:Date|Publication Date|Created)\s*:\s*(\d{4}[-/]\d{2}[-/]\d{2}|\d{4})',
]

CONTACT_PATTERNS = {
    'name': [
        r'(?:contact|point\s*of\s*contact|poc)\s*(?:name)?\s*[:=]\s*(.+)',
        r'(?:author|creator|investigator)\s*[:=]\s*(.+)',
    ],
    'email': [
        r'(?:email|e-mail|contact\s*email)\s*[:=]\s*([^\s,;]+@[^\s,;]+)',
        r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
    ],
}

FORMAT_PATTERNS = [
    r'(?:format|data\s*format|file\s*type)\s*[:=]\s*(.+)',
    r'Format\s*:\s*(.+)',
]

URL_PATTERNS = [
    r'(?:url|link|access|download)\s*[:=]\s*(https?://[^\s]+)',
    r'(https?://[^\s]+)',
]


def extract_with_patterns(text: str, patterns: list, flags=re.IGNORECASE | re.MULTILINE) -> Optional[str]:
    """Try multiple regex patterns and return the first match."""
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1).strip()
    return None


def parse_txt_metadata(text: str) -> dict:
    """Parse free-text metadata into a normalized dict using heuristics."""
    result = {}

    # Title
    title = extract_with_patterns(text, TITLE_PATTERNS)
    if title and len(title) < 300:
        result['title'] = title
    else:
        # Fallback: use first non-empty line
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 5 and len(line) < 300:
                result['title'] = line
                break

    # Abstract/Notes
    abstract = extract_with_patterns(text, ABSTRACT_PATTERNS)
    if abstract:
        result['notes'] = abstract[:2000]  # Truncate very long abstracts
    else:
        # Use first substantial paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        if paragraphs:
            result['notes'] = paragraphs[0][:2000]

    # Keywords
    kw_str = extract_with_patterns(text, KEYWORD_PATTERNS)
    if kw_str:
        # Split on common delimiters
        keywords = re.split(r'[,;|]+', kw_str)
        result['tags'] = [k.strip() for k in keywords if k.strip()]

    # Date
    date = extract_with_patterns(text, DATE_PATTERNS)
    if date:
        # Normalize format
        date_normalized = re.sub(r'/', '-', date)
        result['extras.issueDate'] = date_normalized
        result['extras.lastUpdateDate'] = date_normalized

    # Contact Name
    name = extract_with_patterns(text, CONTACT_PATTERNS['name'])
    if name and len(name) < 100:
        result['extras.pocName'] = name

    # Contact Email
    email = extract_with_patterns(text, CONTACT_PATTERNS['email'])
    if email:
        result['extras.pocEmail'] = email

    # Format
    fmt = extract_with_patterns(text, FORMAT_PATTERNS)
    if fmt and len(fmt) < 50:
        result['resource.format'] = fmt.upper()
        mime_map = {
            'CSV': 'text/csv',
            'JSON': 'application/json',
            'XML': 'text/xml',
            'PDF': 'application/pdf',
            'TXT': 'text/plain',
            'NETCDF': 'application/x-netcdf',
            'NC': 'application/x-netcdf',
            'ZIP': 'application/zip',
        }
        result['resource.mimetype'] = mime_map.get(fmt.upper().strip(), '')

    # URL
    url = extract_with_patterns(text, URL_PATTERNS)
    if url:
        result['resource.url'] = url

    result['extras.uploadType'] = 'dataset'
    result['extras.dataType'] = 'tabular'
    result['source_format'] = 'TXT'
    return result


def clean_text(text: str) -> str:
    """Clean raw text for processing."""
    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    # Remove null bytes
    text = text.replace('\x00', '')
    return text.strip()
