from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "3"))  # extract first N pages
    output_dir: str = os.getenv("OUTPUT_DIR", "output")

settings = Settings()