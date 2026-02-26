# PDF → Metadata JSON Agent (Streamlit + Ollama)

## Prereqs
1) Install and run Ollama
- `ollama serve`
- `ollama pull llama3`

2) Create venv + install deps
- `python -m venv .venv`
- `source .venv/bin/activate` (mac/linux)
- `pip install -r requirements.txt`

## Run
- `streamlit run app.py`

## Notes
- The app extracts PDF document metadata + first N pages of text
- Llama3 converts extracted facts to CKAN-like JSON
- JSON is validated using a simple schema (expand as needed)