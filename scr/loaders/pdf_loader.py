"""
PDF Loader: Extracts metadata from PDF files.
"""
import io
import re
from typing import Optional

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


def extract_pdf_text(pdf_bytes: bytes, max_pages: int = 10) -> str:
    """Extract text from a PDF file."""
    text_parts = []

    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages[:max_pages]):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception:
            pass

    if HAS_PYPDF:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for i, page in enumerate(reader.pages[:max_pages]):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception:
            pass

    return ""


def extract_pdf_metadata(pdf_bytes: bytes) -> dict:
    """Extract document-level metadata from a PDF."""
    meta = {}

    if HAS_PYPDF:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            info = reader.metadata
            if info:
                if info.get('/Title'):
                    meta['pdf_title'] = str(info['/Title'])
                if info.get('/Author'):
                    meta['pdf_author'] = str(info['/Author'])
                if info.get('/Subject'):
                    meta['pdf_subject'] = str(info['/Subject'])
                if info.get('/Keywords'):
                    meta['pdf_keywords'] = str(info['/Keywords'])
                if info.get('/CreationDate'):
                    meta['pdf_creation_date'] = str(info['/CreationDate'])
            meta['page_count'] = len(reader.pages)
        except Exception:
            pass

    return meta


def parse_pdf_metadata(pdf_bytes: bytes, filename: str = "", max_pages: int = 5) -> dict:
    """Parse a PDF file and extract metadata fields."""
    from loaders.text_loader import parse_txt_metadata, clean_text

    result = {}

    # Extract document metadata first
    doc_meta = extract_pdf_metadata(pdf_bytes)

    # Use PDF-level title/author if available
    if doc_meta.get('pdf_title') and len(doc_meta['pdf_title']) > 3:
        result['title'] = doc_meta['pdf_title']

    if doc_meta.get('pdf_author'):
        result['extras.pocName'] = doc_meta['pdf_author']

    if doc_meta.get('pdf_keywords'):
        kws = re.split(r'[,;]+', doc_meta['pdf_keywords'])
        result['tags'] = [k.strip() for k in kws if k.strip()]

    # Extract text and run heuristic parser
    text = extract_pdf_text(pdf_bytes, max_pages=max_pages)
    if text:
        text_clean = clean_text(text)
        text_result = parse_txt_metadata(text_clean)

        # Merge: text_result overrides only if we don't have a value yet
        for k, v in text_result.items():
            if k not in result or not result[k]:
                result[k] = v

        # Store raw text for LLM processing
        result['_raw_text'] = text_clean[:3000]

    # Use filename as fallback title
    if not result.get('title') and filename:
        name = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        result['title'] = name.title()

    result['extras.uploadType'] = 'dataset'
    result['source_format'] = 'PDF'
    result['page_count'] = doc_meta.get('page_count', 0)
    return result
