from typing import Dict, Any
from lxml import etree
import xmltodict
from .base import BaseLoader

class XmlLoader(BaseLoader):
    def load(self, path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            raw = f.read()

        # 1) keep raw text (sometimes useful for LLM)
        try:
            root = etree.fromstring(raw)
            root_tag = root.tag
        except Exception:
            root_tag = None

        # 2) also provide dict form (good for deterministic extraction)
        try:
            obj = xmltodict.parse(raw)
        except Exception:
            obj = None

        # Heuristic: treat as EML if root tag contains "eml" or known namespace
        inferred = "eml" if (root_tag and "eml" in root_tag.lower()) else "xml"

        return {
            "input_type": inferred,   # "xml" or "eml"
            "root_tag": root_tag,
            "xml_as_dict": obj,
            "xml_raw": raw.decode("utf-8", errors="ignore"),
        }
