from typing import Dict, Any
from .base import BaseLoader

class TextLoader(BaseLoader):
    def load(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return {"input_type": "txt", "text": text}
