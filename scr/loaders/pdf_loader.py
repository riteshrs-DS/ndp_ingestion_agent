from typing import Dict, Any

from pypdf import PdfReader
from .base import BaseLoader

class PdfLoader(BaseLoader):
    def __init__(self, max_pages: int = 3):
        self.max_pages = max_pages

    def load(self, path: str) -> Dict[str, Any]:
        reader = PdfReader(path)
        meta = {}
        if reader.metadata:
            for k, v in reader.metadata.items():
                meta[str(k)] = str(v)

        pages_text = []
        n = min(len(reader.pages), self.max_pages)
        for i in range(n):
            pages_text.append((reader.pages[i].extract_text() or "").strip())

        return {"input_type": "pdf", "pdf_metadata": meta, "pages_text": pages_text}
