from dataclasses import dataclass
from typing import Dict, Any, List
from pypdf import PdfReader

@dataclass
class PdfFacts:
    meta: Dict[str, Any]
    pages_text: List[str]

def extract_pdf_facts(pdf_path: str, max_pages: int = 3) -> PdfFacts:
    reader = PdfReader(pdf_path)

    meta = {}
    if reader.metadata:
        # pypdf returns a DocumentInformation-like object
        for k, v in reader.metadata.items():
            meta[str(k)] = str(v)

    pages_text: List[str] = []
    n = min(len(reader.pages), max_pages)
    for i in range(n):
        text = reader.pages[i].extract_text() or ""
        pages_text.append(text.strip())

    return PdfFacts(meta=meta, pages_text=pages_text)