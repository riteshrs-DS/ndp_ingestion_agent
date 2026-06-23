"""
eval_engine.py  ·  NDP Ingestion Agent — System Evaluation Engine
──────────────────────────────────────────────────────────────────
Evaluates the extraction pipeline across three dimensions:

  1. Accuracy   — field-level precision, completeness score,
                  format validity, value quality checks
  2. Timing     — wall-clock extraction time per loader and per field
  3. Tokens     — prompt tokens, response tokens, cost estimate
                  for every LLM call made during enrichment

Designed to be run both:
  • Standalone (python eval/eval_engine.py) for batch CI evaluation
  • From Streamlit (tab "📊 Evaluation") for interactive reporting
"""

import re
import sys
import os
import time
import json
import hashlib
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict

# Support both: python eval/eval_engine.py  AND  import from app
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.schema import PUBLIC_REQUIRED_FIELDS, PRIVATE_REQUIRED_FIELDS, VALID_UPLOAD_TYPES, VALID_RESOURCE_STATUSES
from utils.validator import preflight_check, build_ckan_package
from loaders.text_loader import parse_txt_metadata, clean_text
from loaders.mlcommons_loader import normalize_mlcommons_to_flat, MLCOMMONS_CATALOG
from loaders.huggingface_loader import popular_dataset_to_flat, POPULAR_HF_DATASETS
from loaders.xml_loader import parse_iso_xml, parse_eml_xml, normalize_from_ckan_api

# ── Token estimation (no external network needed) ──────────────────────────────
# cl100k_base approximation: English prose ~4 chars/token, code/JSON ~3.5 chars/token
def estimate_tokens(text: str) -> int:
    """
    Estimate token count without network calls.
    Uses character-based heuristic calibrated against cl100k_base:
      - JSON / structured: 3.5 chars per token
      - English prose: 4.0 chars per token
    Returns an integer token count.
    """
    if not text:
        return 0
    is_structured = text.strip().startswith(('{', '[', '<'))
    chars_per_token = 3.5 if is_structured else 4.0
    return max(1, int(len(text) / chars_per_token))


# Token cost table (USD per 1K tokens, as of mid-2025)
# Input cost / Output cost
TOKEN_COSTS = {
    "ollama/llama3":          (0.0,    0.0),
    "ollama/llama2":          (0.0,    0.0),
    "ollama/mistral":         (0.0,    0.0),
    "ellm/gemma":             (0.0002, 0.0002),
    "ellm/qwen3":             (0.0003, 0.0003),
    "ellm/glm-v":             (0.0003, 0.0003),
    "ellm/gpt-oss":           (0.0005, 0.0005),
    "ellm/kimi":              (0.0003, 0.0003),
    "ellm/glm-4.7":           (0.0004, 0.0004),
    "ellm/minimax-m2":        (0.0004, 0.0004),
    "anthropic/claude-sonnet":(0.003,  0.015),
    "openai/gpt-4o-mini":     (0.00015,0.0006),
}


# ── Ground-truth fixtures for accuracy testing ────────────────────────────────

GROUND_TRUTH_FIXTURES = [
    {
        "id": "txt_hpwren",
        "source": "TXT",
        "loader": "text_loader",
        "input": """Title: HPWREN Weather Station Measurements
Description: Archive of HPWREN weather station measurements from 2007-present. Includes SDG&E weather station measurements until July 2018.
Keywords: weather, sensors, San Diego, timeseries, HPWREN
Date: 2020-03-06
Contact: Hans-Werner Braun
Email: hwb@ucsd.edu
Format: CSV
URL: https://ndp-test.sdsc.edu/catalog/dataset/hpwren
""",
        "expected": {
            "title": "HPWREN Weather Station Measurements",
            "notes": "Archive of HPWREN weather station measurements from 2007-present",
            "tags": ["weather", "sensors", "San Diego", "timeseries", "HPWREN"],
            "extras.pocName": "Hans-Werner Braun",
            "extras.pocEmail": "hwb@ucsd.edu",
            "extras.issueDate": "2020-03-06",
            "resource.format": "CSV",
            "resource.mimetype": "text/csv",
            "extras.uploadType": "dataset",
        },
    },
    {
        "id": "txt_climate",
        "source": "TXT",
        "loader": "text_loader",
        "input": """Dataset Name: CLM Climate Model Output
Abstract: Long-term climate model simulation outputs from the Community Land Model. Contains daily temperature, precipitation, and soil moisture data for the western United States from 1980-2020.
Keywords: climate, land model, temperature, precipitation, soil, western US
Publication Date: 2019-07-15
Author: Dr. Sarah Johnson
Email: sjohnson@climate.edu
Format: NetCDF
URL: https://ndp-test.sdsc.edu/catalog/dataset/clm_output
""",
        "expected": {
            "title": "CLM Climate Model Output",
            "notes": "Long-term climate model simulation outputs",
            "tags": ["climate", "land model", "temperature", "precipitation"],
            "extras.pocName": "Dr. Sarah Johnson",
            "extras.pocEmail": "sjohnson@climate.edu",
            "extras.issueDate": "2019-07-15",
            "resource.format": "NETCDF",
            "extras.uploadType": "dataset",
        },
    },
    {
        "id": "txt_wildfire",
        "source": "TXT",
        "loader": "text_loader",
        "input": """Title: California Wildfire Treatment Dashboard Data
Description: Statewide tracking of wildfire risk reduction treatments across California landscapes. Data includes treatment type, acreage, completion status, and geographic extent from 2021-2024.
Tags: wildfire, fire risk, treatment, California, forestry, landscape
Created: 2022-01-01
Contact: CAL FIRE Data Team
Email: data@fire.ca.gov
Format: CSV
URL: https://wildfiretaskforce.org/treatment-dashboard/
""",
        "expected": {
            "title": "California Wildfire Treatment Dashboard Data",
            "tags": ["wildfire", "fire risk", "treatment", "California", "forestry"],
            "extras.pocName": "CAL FIRE Data Team",
            "extras.pocEmail": "data@fire.ca.gov",
            "extras.issueDate": "2022-01-01",
            "resource.format": "CSV",
            "extras.uploadType": "dataset",
        },
    },
    {
        "id": "mlc_peoples_speech",
        "source": "MLCommons",
        "loader": "mlcommons_loader",
        "input": "peoples-speech",   # slug
        "expected": {
            "title": "People's Speech",
            "extras.uploadType": "dataset",
            "extras.dataType": "audio",
            "extras.pocName": "MLCommons",
            "extras.pocEmail": "datasets@mlcommons.org",
            "extras.license": "CC-BY-SA 4.0",
            "resource.format": "FLAC",
            "resource.mimetype": "audio/flac",
            "resource.status": "active",
        },
    },
    {
        "id": "mlc_dollar_street",
        "source": "MLCommons",
        "loader": "mlcommons_loader",
        "input": "dollar-street",
        "expected": {
            "title": "Dollar Street",
            "extras.uploadType": "dataset",
            "extras.dataType": "imagery",
            "extras.license": "CC-BY / CC-BY-SA 4.0",
            "resource.format": "JPG",
            "resource.mimetype": "image/jpeg",
            "resource.status": "active",
        },
    },
    {
        "id": "hf_squad",
        "source": "HuggingFace",
        "loader": "huggingface_loader",
        "input": "squad",            # popular catalog entry id
        "expected": {
            "title": "SQuAD",
            "extras.uploadType": "dataset",
            "extras.dataType": "text",
            "extras.license": "cc-by-sa-4.0",
            "resource.format": "Parquet",
            "resource.mimetype": "application/parquet",
            "resource.status": "active",
        },
    },
    {
        "id": "hf_imdb",
        "source": "HuggingFace",
        "loader": "huggingface_loader",
        "input": "imdb",
        "expected": {
            "title": "IMDB Reviews",
            "extras.uploadType": "dataset",
            "extras.dataType": "text",
            "extras.license": "apache-2.0",
            "resource.format": "Parquet",
            "resource.mimetype": "application/parquet",
        },
    },
    {
        "id": "iso_xml_minimal",
        "source": "ISO/XML",
        "loader": "xml_loader",
        "input": """<?xml version="1.0" encoding="UTF-8"?>
<gmd:MD_Metadata xmlns:gmd="http://www.isotc211.org/2005/gmd"
                 xmlns:gco="http://www.isotc211.org/2005/gco">
  <gmd:identificationInfo>
    <gmd:MD_DataIdentification>
      <gmd:citation>
        <gmd:CI_Citation>
          <gmd:title><gco:CharacterString>BCO Weather Station Sensor Data</gco:CharacterString></gmd:title>
        </gmd:CI_Citation>
      </gmd:citation>
      <gmd:abstract><gco:CharacterString>High-resolution meteorological sensor data from the BCO weather buoy network in the Atlantic Ocean, 2015-2023.</gco:CharacterString></gmd:abstract>
      <gmd:descriptiveKeywords>
        <gmd:MD_Keywords>
          <gmd:keyword><gco:CharacterString>oceanography</gco:CharacterString></gmd:keyword>
          <gmd:keyword><gco:CharacterString>meteorology</gco:CharacterString></gmd:keyword>
          <gmd:keyword><gco:CharacterString>sensors</gco:CharacterString></gmd:keyword>
        </gmd:MD_Keywords>
      </gmd:descriptiveKeywords>
      <gmd:pointOfContact>
        <gmd:CI_ResponsibleParty>
          <gmd:individualName><gco:CharacterString>Dr. Maria Santos</gco:CharacterString></gmd:individualName>
          <gmd:contactInfo>
            <gmd:CI_Contact>
              <gmd:address>
                <gmd:CI_Address>
                  <gmd:electronicMailAddress><gco:CharacterString>msantos@bco.de</gco:CharacterString></gmd:electronicMailAddress>
                </gmd:CI_Address>
              </gmd:address>
            </gmd:CI_Contact>
          </gmd:contactInfo>
        </gmd:CI_ResponsibleParty>
      </gmd:pointOfContact>
    </gmd:MD_DataIdentification>
  </gmd:identificationInfo>
  <gmd:dateStamp><gco:Date>2023-06-15</gco:Date></gmd:dateStamp>
</gmd:MD_Metadata>""",
        "expected": {
            "title": "BCO Weather Station Sensor Data",
            "notes": "High-resolution meteorological sensor data",
            "tags": ["oceanography", "meteorology", "sensors"],
            "extras.pocName": "Dr. Maria Santos",
            "extras.pocEmail": "msantos@bco.de",
            "extras.issueDate": "2023-06-15",
            "extras.uploadType": "dataset",
        },
    },
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FieldResult:
    field: str
    expected: str
    got: str
    present: bool          # field was extracted at all
    exact_match: bool      # exact string equality
    partial_match: bool    # expected is substring of got (or vice versa)
    format_valid: bool     # passes format validation rules
    score: float           # 0.0 – 1.0


@dataclass
class TokenRecord:
    call_type: str          # "normalize" | "repair" | "generate"
    model_key: str
    prompt_text: str
    response_text: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int
    cost_usd: float


@dataclass
class ExtractionResult:
    fixture_id: str
    source_type: str
    loader_name: str
    # Timing
    extraction_time_ms: float
    # Accuracy
    field_results: list[FieldResult]
    fields_required: int
    fields_present: int
    fields_exact: int
    fields_partial: int
    completeness_pct: float
    accuracy_pct: float        # exact / required
    overall_score: float       # weighted: 0.6*accuracy + 0.4*completeness
    preflight_passed: bool
    missing_required: list[str]
    format_errors: list[str]
    # Tokens (populated only when LLM enrichment is tested)
    token_records: list[TokenRecord] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_response_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    # Raw output
    extracted_flat: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── Field-level accuracy scoring ─────────────────────────────────────────────

def _check_format(field_name: str, value: str) -> tuple[bool, str]:
    """Return (valid, error_message) for known format-constrained fields."""
    if not value:
        return True, ""   # missing is handled separately

    if field_name == "extras.pocEmail":
        if "@" not in value or "." not in value.split("@")[-1]:
            return False, f"'{value}' does not look like a valid email"

    if field_name in ("extras.issueDate", "extras.lastUpdateDate"):
        if not re.match(r'^\d{4}-\d{2}-\d{2}', value):
            return False, f"'{value}' is not in YYYY-MM-DD format"

    if field_name == "extras.uploadType":
        if value.lower() not in VALID_UPLOAD_TYPES:
            return False, f"'{value}' not in {VALID_UPLOAD_TYPES}"

    if field_name == "resource.status":
        if value.lower() not in VALID_RESOURCE_STATUSES:
            return False, f"'{value}' not in {VALID_RESOURCE_STATUSES}"

    if field_name == "resource.mimetype":
        if "/" not in value:
            return False, f"'{value}' does not look like a MIME type (missing '/')"

    return True, ""


def _score_field(field_name: str, expected: str, got) -> FieldResult:
    """Score one field extraction against ground truth."""
    # Normalise both sides
    expected_s = str(expected).lower().strip() if expected is not None else ""
    got_s = ""
    present = False

    if isinstance(got, list):
        present = len(got) > 0
        got_s = " ".join(str(g).lower().strip() for g in got)
    elif got is not None and str(got).strip():
        present = True
        got_s = str(got).lower().strip()

    exact = (got_s == expected_s) if present else False

    # Partial: either direction containment, or 60% word overlap
    partial = False
    if present and expected_s:
        if expected_s in got_s or got_s in expected_s:
            partial = True
        else:
            exp_words = set(expected_s.split())
            got_words = set(got_s.split())
            if exp_words and got_words:
                overlap = len(exp_words & got_words) / len(exp_words)
                partial = overlap >= 0.6

    fmt_valid, _ = _check_format(field_name, str(got) if got else "")

    # Score: 1.0 exact, 0.6 partial+format_valid, 0.4 partial,
    #         0.3 present+format_valid, 0.1 present, 0.0 missing
    if exact and fmt_valid:
        score = 1.0
    elif partial and fmt_valid:
        score = 0.6
    elif partial:
        score = 0.4
    elif present and fmt_valid:
        score = 0.3
    elif present:
        score = 0.1
    else:
        score = 0.0

    return FieldResult(
        field=field_name,
        expected=str(expected),
        got=str(got) if got else "",
        present=present,
        exact_match=exact,
        partial_match=partial,
        format_valid=fmt_valid,
        score=score,
    )


# ── Loaders ───────────────────────────────────────────────────────────────────

def _run_loader(fixture: dict) -> tuple[dict, float]:
    """Run the appropriate loader and return (flat_metadata, elapsed_ms)."""
    loader = fixture["loader"]
    inp = fixture["input"]

    t0 = time.perf_counter()

    if loader == "text_loader":
        result = parse_txt_metadata(clean_text(inp))

    elif loader == "xml_loader":
        if "eml" in inp.lower()[:100] or "ecoinformatics" in inp.lower()[:100]:
            result = parse_eml_xml(inp)
        else:
            result = parse_iso_xml(inp)

    elif loader == "mlcommons_loader":
        entry = normalize_mlcommons_to_flat(
            next(d for d in MLCOMMONS_CATALOG if d["slug"] == inp)
        )
        result = entry

    elif loader == "huggingface_loader":
        entry = next((d for d in POPULAR_HF_DATASETS if d["id"] == inp), None)
        result = popular_dataset_to_flat(entry) if entry else {}

    else:
        result = {}

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


# ── Core evaluator ────────────────────────────────────────────────────────────

def evaluate_fixture(fixture: dict) -> ExtractionResult:
    """Run full evaluation for one fixture (accuracy + timing)."""
    flat, elapsed_ms = _run_loader(fixture)
    expected = fixture["expected"]

    # Field-level scoring
    field_results = []
    format_errors = []
    for fname, exp_val in expected.items():
        got_val = flat.get(fname)
        fr = _score_field(fname, exp_val, got_val)
        field_results.append(fr)
        if not fr.format_valid and fr.present:
            fmt_ok, err_msg = _check_format(fname, str(got_val))
            if err_msg:
                format_errors.append(f"{fname}: {err_msg}")

    # Aggregate stats
    required = set(PUBLIC_REQUIRED_FIELDS.keys())
    evaluated_required = [fr for fr in field_results if fr.field in required]

    fields_required  = len(evaluated_required)
    fields_present   = sum(1 for fr in evaluated_required if fr.present)
    fields_exact     = sum(1 for fr in evaluated_required if fr.exact_match)
    fields_partial   = sum(1 for fr in evaluated_required if fr.partial_match and not fr.exact_match)

    completeness_pct = (fields_present / fields_required * 100) if fields_required else 0
    accuracy_pct     = (fields_exact   / fields_required * 100) if fields_required else 0

    # Weighted overall score over ALL evaluated fields
    avg_score = sum(fr.score for fr in field_results) / len(field_results) if field_results else 0
    overall_score = avg_score * 100

    # Preflight
    ckan_pkg = build_ckan_package(flat)
    pfr = preflight_check(flat)

    return ExtractionResult(
        fixture_id=fixture["id"],
        source_type=fixture["source"],
        loader_name=fixture["loader"],
        extraction_time_ms=round(elapsed_ms, 2),
        field_results=field_results,
        fields_required=fields_required,
        fields_present=fields_present,
        fields_exact=fields_exact,
        fields_partial=fields_partial,
        completeness_pct=round(completeness_pct, 1),
        accuracy_pct=round(accuracy_pct, 1),
        overall_score=round(overall_score, 1),
        preflight_passed=pfr["passed"],
        missing_required=pfr["missing_fields"],
        format_errors=format_errors,
        extracted_flat=flat,
    )


def run_all_fixtures(
    progress_cb: Optional[Callable[[str, float], None]] = None
) -> list[ExtractionResult]:
    """Evaluate all ground-truth fixtures. Returns list of ExtractionResult."""
    results = []
    total = len(GROUND_TRUTH_FIXTURES)
    for i, fixture in enumerate(GROUND_TRUTH_FIXTURES):
        if progress_cb:
            progress_cb(f"Evaluating [{fixture['id']}]…", (i / total))
        results.append(evaluate_fixture(fixture))
    if progress_cb:
        progress_cb("Complete", 1.0)
    return results


# ── Token accounting for LLM calls ───────────────────────────────────────────

def build_token_record(
    call_type: str,
    model_key: str,
    prompt: str,
    response: str,
) -> TokenRecord:
    prompt_tok   = estimate_tokens(prompt)
    response_tok = estimate_tokens(response)
    total        = prompt_tok + response_tok
    in_cost, out_cost = TOKEN_COSTS.get(model_key, (0.0, 0.0))
    cost = (prompt_tok / 1000 * in_cost) + (response_tok / 1000 * out_cost)
    return TokenRecord(
        call_type=call_type,
        model_key=model_key,
        prompt_text=prompt[:300] + "…" if len(prompt) > 300 else prompt,
        response_text=response[:300] + "…" if len(response) > 300 else response,
        prompt_tokens=prompt_tok,
        response_tokens=response_tok,
        total_tokens=total,
        cost_usd=round(cost, 6),
    )


def simulate_llm_token_scenarios(model_key: str = "ollama/llama3") -> list[TokenRecord]:
    """
    Simulate the 3 standard LLM calls the agent makes per dataset
    with realistic prompt/response sizes to show token costs.
    """
    from utils.llm_provider import build_normalization_prompt, build_repair_prompt

    # Simulate a TXT metadata dict with partial fields
    sample_meta = {
        "title": "HPWREN Weather Station Measurements",
        "notes": "Archive of HPWREN weather station measurements from 2007-present.",
        "tags": ["weather", "sensors", "San Diego"],
        "extras.issueDate": "2020-03-06",
        "extras.uploadType": "dataset",
        "source_format": "TXT",
        "_raw_text": "Archive of HPWREN weather station measurements from 2007-present. "
                     "Includes SDG&E weather station measurements until July 2018. "
                     "Data includes temperature, humidity, wind speed and direction.",
    }
    missing_fields = ["extras.pocName", "extras.pocEmail", "extras.dataType",
                      "extras.lastUpdateDate", "resource.name", "resource.description",
                      "resource.mimetype", "resource.format", "resource.status"]

    norm_prompt = build_normalization_prompt(sample_meta, missing_fields)
    norm_response = json.dumps({
        "extras.pocName": "Hans-Werner Braun",
        "extras.pocEmail": "hwb@ucsd.edu",
        "extras.dataType": "timeseries",
        "extras.lastUpdateDate": "2020-03-06",
        "resource.name": "HPWREN Weather Data",
        "resource.description": "CSV files with hourly weather station readings.",
        "resource.mimetype": "text/csv",
        "resource.format": "CSV",
        "resource.status": "active",
    }, indent=2)

    repair_pkg = {
        "title": "HPWREN Weather Station Measurements",
        "notes": "Archive of HPWREN measurements",
        "tags": [{"name": "weather"}],
        "extras": [{"key": "uploadType", "value": "dataset"}],
        "resources": [{"name": "", "url": "https://example.com"}]
    }
    repair_errors = ["resource.description is missing", "resource.mimetype is missing"]
    repair_prompt = build_repair_prompt(repair_pkg, repair_errors, {
        "resource.description": "CSV weather data",
        "resource.mimetype": "text/csv"
    })
    repair_response = json.dumps(repair_pkg, indent=2)

    question_prompt = (
        "The following required fields are missing from the NDP metadata record. "
        "Please provide values for: extras.pocName, extras.pocEmail, resource.url"
    )
    question_response = (
        "Point of Contact Name: Hans-Werner Braun\n"
        "Point of Contact Email: hwb@ucsd.edu\n"
        "Resource URL: https://ndp-test.sdsc.edu/catalog/dataset/hpwren"
    )

    return [
        build_token_record("normalize", model_key, norm_prompt, norm_response),
        build_token_record("repair",    model_key, repair_prompt, repair_response),
        build_token_record("questions", model_key, question_prompt, question_response),
    ]


def build_token_summary(
    model_keys: list[str],
) -> list[dict]:
    """
    Build a comparison table of token usage + cost across all registered models
    for a single standard extraction cycle (normalize + repair + questions).
    """
    rows = []
    for key in model_keys:
        records = simulate_llm_token_scenarios(key)
        total_prompt   = sum(r.prompt_tokens for r in records)
        total_response = sum(r.response_tokens for r in records)
        total_tokens   = sum(r.total_tokens for r in records)
        total_cost     = sum(r.cost_usd for r in records)
        in_cost, out_cost = TOKEN_COSTS.get(key, (0.0, 0.0))
        rows.append({
            "model_key":       key,
            "prompt_tokens":   total_prompt,
            "response_tokens": total_response,
            "total_tokens":    total_tokens,
            "cost_per_call_usd": round(total_cost, 6),
            "cost_per_100_usd":  round(total_cost * 100, 4),
            "input_rate":      f"${in_cost}/1K" if in_cost else "free",
            "output_rate":     f"${out_cost}/1K" if out_cost else "free",
        })
    return rows


# ── Summary statistics ────────────────────────────────────────────────────────

def summarise(results: list[ExtractionResult]) -> dict:
    """Aggregate statistics across all evaluation results."""
    if not results:
        return {}

    total = len(results)
    by_source = {}
    for r in results:
        by_source.setdefault(r.source_type, []).append(r)

    return {
        "total_fixtures":      total,
        "avg_completeness_pct": round(sum(r.completeness_pct for r in results) / total, 1),
        "avg_accuracy_pct":    round(sum(r.accuracy_pct for r in results) / total, 1),
        "avg_overall_score":   round(sum(r.overall_score for r in results) / total, 1),
        "avg_time_ms":         round(sum(r.extraction_time_ms for r in results) / total, 2),
        "max_time_ms":         round(max(r.extraction_time_ms for r in results), 2),
        "min_time_ms":         round(min(r.extraction_time_ms for r in results), 2),
        "preflight_pass_rate": round(sum(1 for r in results if r.preflight_passed) / total * 100, 1),
        "total_format_errors": sum(len(r.format_errors) for r in results),
        "by_source": {
            src: {
                "count": len(rs),
                "avg_completeness": round(sum(r.completeness_pct for r in rs) / len(rs), 1),
                "avg_accuracy":     round(sum(r.accuracy_pct for r in rs) / len(rs), 1),
                "avg_time_ms":      round(sum(r.extraction_time_ms for r in rs) / len(rs), 2),
            }
            for src, rs in by_source.items()
        }
    }


# ── CLI runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os; os.chdir(os.path.dirname(__file__) + "/..")

    print("\n" + "="*70)
    print("  NDP Ingestion Agent — System Evaluation")
    print("="*70)

    results = run_all_fixtures(
        progress_cb=lambda msg, pct: print(f"  [{int(pct*100):3d}%] {msg}")
    )

    print("\n── ACCURACY + TIMING ──────────────────────────────────────────")
    header = f"{'ID':<25} {'Source':<12} {'Complete%':>9} {'Exact%':>7} {'Score':>6} {'TimeMs':>7} {'Preflt':>6}"
    print(header)
    print("-" * 80)
    for r in results:
        pf = "✓" if r.preflight_passed else "✗"
        print(f"{r.fixture_id:<25} {r.source_type:<12} {r.completeness_pct:>8.1f}% "
              f"{r.accuracy_pct:>6.1f}% {r.overall_score:>5.1f} {r.extraction_time_ms:>7.1f} {pf:>6}")

    summary = summarise(results)
    print(f"\nAverages: completeness={summary['avg_completeness_pct']}%  "
          f"accuracy={summary['avg_accuracy_pct']}%  "
          f"score={summary['avg_overall_score']}  "
          f"time={summary['avg_time_ms']}ms")

    print("\n── TOKEN & COST COMPARISON (per extraction cycle) ─────────────")
    from utils.llm_registry import MODEL_REGISTRY
    keys = [m["key"] for m in MODEL_REGISTRY]
    rows = build_token_summary(keys)
    hdr2 = f"{'Model':<35} {'Prompt':>8} {'Response':>9} {'Total':>7} {'Cost/call':>10} {'Cost/100':>10}"
    print(hdr2)
    print("-" * 85)
    for row in rows:
        print(f"{row['model_key']:<35} {row['prompt_tokens']:>8} {row['response_tokens']:>9} "
              f"{row['total_tokens']:>7} ${row['cost_per_call_usd']:>9.6f} ${row['cost_per_100_usd']:>9.4f}")
