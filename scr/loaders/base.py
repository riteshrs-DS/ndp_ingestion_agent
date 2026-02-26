from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]:
        """Return normalized facts dict."""
        raise NotImplementedError
