"""
llm_registry.py  ·  Unified LLM Provider Registry
────────────────────────────────────────────────────
Manages every LLM the NDP agent can use:

  Group A – Ollama (local)
      llama3, llama2, mistral  →  http://localhost:11434

  Group B – NRP / ELLM (https://ellm.nrp-nautilus.io)
      gemma, qwen3, glm-v, gpt-oss, kimi, glm-4.7, minimax-m2
      All share the OpenAI-compatible /v1/chat/completions endpoint.
      Auth: ELLM_API_KEY env-var (set once, used by all NRP models).

  Group C – Cloud APIs (OpenAI-compatible)
      anthropic/claude-sonnet  →  Anthropic API
      openai/gpt-4o-mini       →  OpenAI API

All groups expose an identical call interface:
    generate(prompt, model_key, **kwargs) → str | None

Credentials are loaded from (in priority order):
  1. Streamlit session_state  (set at runtime via the sidebar)
  2. Environment variables     (ELLM_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)
  3. .env file in the project root
"""

import os
import re
import json
import requests
from typing import Optional
from dotenv import load_dotenv

# Load .env once at import time
load_dotenv(override=False)

# ── Model catalogue ────────────────────────────────────────────────────────────

OLLAMA_BASE   = "http://localhost:11434"
ELLM_BASE     = "https://ellm.nrp-nautilus.io/v1"
#ANTHROPIC_BASE= "https://api.anthropic.com"
#OPENAI_BASE   = "https://api.openai.com/v1"

# Each entry:
#   key          – unique identifier used everywhere in the app
#   label        – human-readable name shown in the UI
#   group        – "ollama" | "ellm" | "anthropic" | "openai"
#   model        – model name sent to the API
#   base_url     – API base URL
#   api_key_env  – env-var name that holds the credential (None → no auth)
#   notes        – short info string for the UI

MODEL_REGISTRY: list[dict] = [
    # ── Ollama (local) ────────────────────────────────────────────────────────
    {
        "key": "ollama/llama3",
        "label": "LLaMA 3 (Ollama local)",
        "group": "ollama",
        "model": "llama3",
        "base_url": OLLAMA_BASE,
        "api_key_env": None,
        "notes": "Best all-round local model. Requires Ollama running.",
    },
    {
        "key": "ollama/llama2",
        "label": "LLaMA 2 (Ollama local)",
        "group": "ollama",
        "model": "llama2",
        "base_url": OLLAMA_BASE,
        "api_key_env": None,
        "notes": "Lighter local model. Requires Ollama running.",
    },
    {
        "key": "ollama/mistral",
        "label": "Mistral 7B (Ollama local)",
        "group": "ollama",
        "model": "mistral",
        "base_url": OLLAMA_BASE,
        "api_key_env": None,
        "notes": "Fast local model. Requires Ollama running.",
    },
    # ── NRP / ELLM ────────────────────────────────────────────────────────────
    {
        "key": "ellm/gemma",
        "label": "Gemma (NRP-ELLM)",
        "group": "ellm",
        "model": "gemma",
        "base_url": ELLM_BASE,
        "api_key_env": "ELLM_API_KEY",
        "notes": "Google Gemma via NRP-Nautilus. Needs ELLM_API_KEY.",
    },
    {
        "key": "ellm/qwen3",
        "label": "Qwen 3 (NRP-ELLM)",
        "group": "ellm",
        "model": "qwen3",
        "base_url": ELLM_BASE,
        "api_key_env": "ELLM_API_KEY",
        "notes": "Alibaba Qwen3 via NRP-Nautilus. Needs ELLM_API_KEY.",
    },
    {
        "key": "ellm/glm-v",
        "label": "GLM-V (NRP-ELLM)",
        "group": "ellm",
        "model": "glm-v",
        "base_url": ELLM_BASE,
        "api_key_env": "ELLM_API_KEY",
        "notes": "Zhipu GLM multimodal via NRP-Nautilus. Needs ELLM_API_KEY.",
    },
    {
        "key": "ellm/gpt-oss",
        "label": "GPT-OSS (NRP-ELLM eval)",
        "group": "ellm",
        "model": "gpt-oss",
        "base_url": ELLM_BASE,
        "api_key_env": "ELLM_API_KEY",
        "notes": "NRP evaluation model. Needs ELLM_API_KEY.",
    },
    {
        "key": "ellm/kimi",
        "label": "Kimi (NRP-ELLM eval)",
        "group": "ellm",
        "model": "kimi",
        "base_url": ELLM_BASE,
        "api_key_env": "ELLM_API_KEY",
        "notes": "Moonshot Kimi via NRP-Nautilus eval. Needs ELLM_API_KEY.",
    },
    {
        "key": "ellm/glm-4.7",
        "label": "GLM-4.7 (NRP-ELLM eval)",
        "group": "ellm",
        "model": "glm-4.7",
        "base_url": ELLM_BASE,
        "api_key_env": "ELLM_API_KEY",
        "notes": "Zhipu GLM-4.7 via NRP-Nautilus eval. Needs ELLM_API_KEY.",
    },
    {
        "key": "ellm/minimax-m2",
        "label": "MiniMax M2 (NRP-ELLM eval)",
        "group": "ellm",
        "model": "minimax-m2",
        "base_url": ELLM_BASE,
        "api_key_env": "ELLM_API_KEY",
        "notes": "MiniMax M2 via NRP-Nautilus eval. Needs ELLM_API_KEY.",
    },
    # ── Anthropic ─────────────────────────────────────────────────────────────
   #{
   #     "key": "anthropic/claude-sonnet",
   #     "label": "Claude Sonnet 4 (Anthropic)",
   #     "group": "anthropic",
   #     "model": "claude-sonnet-4-20250514",
   #     "base_url": ANTHROPIC_BASE,
   #     "api_key_env": "ANTHROPIC_API_KEY",
   #     "notes": "Anthropic Claude via cloud API. Needs ANTHROPIC_API_KEY.",
   # },
    # ── OpenAI ────────────────────────────────────────────────────────────────
  #  {
  #      "key": "openai/gpt-4o-mini",
  #      "label": "GPT-4o Mini (OpenAI)",
  #      "group": "openai",
  #      "model": "gpt-4o-mini",
  #      "base_url": OPENAI_BASE,
  #      "api_key_env": "OPENAI_API_KEY",
  #      "notes": "OpenAI GPT-4o Mini via cloud API. Needs OPENAI_API_KEY.",
  #  },
]

# Fast lookup: key → entry
_REGISTRY_MAP: dict[str, dict] = {m["key"]: m for m in MODEL_REGISTRY}

DEFAULT_MODEL_KEY = "ollama/llama3"

# ── Group metadata (for UI sections) ─────────────────────────────────────────
GROUP_META = {
    "ollama":    {"label": "🖥️  Ollama (Local)",          "color": "#148f77"},
    "ellm":      {"label": "🔬  NRP-ELLM (Nautilus)",     "color": "#1a5276"},
    "anthropic": {"label": "🤖  Anthropic (Cloud)",        "color": "#6c3483"},
    "openai":    {"label": "🌐  OpenAI (Cloud)",           "color": "#117a65"},
}


# ── Credential resolution ─────────────────────────────────────────────────────

def _resolve_api_key(entry: dict, session_overrides: dict = None) -> Optional[str]:
    """
    Return the API key for a model entry.
    Priority: session_overrides → env-var → None
    """
    env_name = entry.get("api_key_env")
    if not env_name:
        return None   # Ollama needs no key

    # 1. Runtime override (from Streamlit sidebar)
    if session_overrides and env_name in session_overrides:
        val = session_overrides[env_name]
        if val and val.strip():
            return val.strip()

    # 2. Environment variable / .env file
    val = os.environ.get(env_name, "")
    return val.strip() if val.strip() else None


def _resolve_base_url(entry: dict, session_overrides: dict = None) -> str:
    """Allow sidebar override of the Ollama base URL."""
    if entry["group"] == "ollama" and session_overrides:
        return session_overrides.get("OLLAMA_BASE_URL", entry["base_url"])
    return entry["base_url"]


# ── Connectivity check ────────────────────────────────────────────────────────

def check_model_connectivity(
    model_key: str,
    session_overrides: dict = None,
) -> tuple[bool, str]:
    """
    Probe whether a model is reachable.
    Returns (ok: bool, message: str).
    """
    entry = _REGISTRY_MAP.get(model_key)
    if not entry:
        return False, f"Unknown model key: {model_key}"

    base_url = _resolve_base_url(entry, session_overrides)
    api_key  = _resolve_api_key(entry, session_overrides)
    group    = entry["group"]

    try:
        if group == "ollama":
            r = requests.get(f"{base_url}/api/tags", timeout=5)
            if r.ok:
                models = [m["name"] for m in r.json().get("models", [])]
                model_name = entry["model"]
                found = any(model_name in m for m in models)
                if found:
                    return True, f"✅ Connected — `{model_name}` is available"
                return True, (
                    f"⚠️ Ollama running but `{model_name}` not pulled. "
                    f"Run: `ollama pull {model_name}`"
                )
            return False, f"❌ Ollama not responding at {base_url}"

        elif group in ("ellm", "openai"):
            if not api_key:
                return False, f"❌ No API key set for {entry['api_key_env']}"
            r = requests.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=8,
            )
            if r.ok:
                return True, f"✅ Connected to {base_url}"
            return False, f"❌ HTTP {r.status_code} from {base_url}"

        elif group == "anthropic":
            if not api_key:
                return False, f"❌ No API key set for {entry['api_key_env']}"
            # Lightweight models list probe
            r = requests.get(
                f"{base_url}/v1/models",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                timeout=8,
            )
            if r.ok:
                return True, "✅ Anthropic API connected"
            return False, f"❌ HTTP {r.status_code} from Anthropic API"

    except requests.exceptions.ConnectionError:
        return False, f"❌ Cannot reach {base_url} (connection refused)"
    except Exception as e:
        return False, f"❌ {type(e).__name__}: {e}"

    return False, "❌ Unknown error"


def check_all_ollama_models(session_overrides: dict = None) -> dict[str, list[str]]:
    """Return dict of available Ollama models pulled on the local instance."""
    base_url = (session_overrides or {}).get("OLLAMA_BASE_URL", OLLAMA_BASE)
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        if r.ok:
            return {"models": [m["name"] for m in r.json().get("models", [])]}
    except Exception:
        pass
    return {"models": []}


# ── Core generate function ────────────────────────────────────────────────────

def generate(
    prompt: str,
    model_key: str = DEFAULT_MODEL_KEY,
    session_overrides: dict = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> Optional[str]:
    """
    Send a prompt to any registered model and return the text response.

    Args:
        prompt           – The full prompt string.
        model_key        – Registry key (e.g. "ollama/llama3", "ellm/qwen3").
        session_overrides– Dict of {env_var_name: value} for runtime credentials.
        temperature      – Sampling temperature (0 = deterministic).
        max_tokens       – Max tokens to generate.

    Returns:
        Response text string, or None on failure.
    """
    entry = _REGISTRY_MAP.get(model_key)
    if not entry:
        raise ValueError(f"Unknown model key '{model_key}'. "
                         f"Valid keys: {list(_REGISTRY_MAP)}")

    base_url = _resolve_base_url(entry, session_overrides)
    api_key  = _resolve_api_key(entry, session_overrides)
    group    = entry["group"]
    model    = entry["model"]

    if group == "ollama":
        return _generate_ollama(prompt, model, base_url, temperature, max_tokens)
    elif group in ("ellm", "openai"):
        return _generate_openai_compat(prompt, model, base_url, api_key,
                                        temperature, max_tokens)
    elif group == "anthropic":
        return _generate_anthropic(prompt, model, api_key, temperature, max_tokens)

    return None


# ── Backend implementations ───────────────────────────────────────────────────

def _generate_ollama(
    prompt: str, model: str, base_url: str,
    temperature: float, max_tokens: int,
) -> Optional[str]:
    """Ollama /api/generate (native endpoint, no auth)."""
    try:
        r = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=120,
        )
        if r.ok:
            return r.json().get("response", "")
    except Exception:
        pass
    return None


def _generate_openai_compat(
    prompt: str, model: str, base_url: str, api_key: Optional[str],
    temperature: float, max_tokens: int,
) -> Optional[str]:
    """
    OpenAI-compatible /v1/chat/completions endpoint.
    Used by: ELLM/NRP models, OpenAI GPT-4o-mini.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        r = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        if r.ok:
            data = r.json()
            return data["choices"][0]["message"]["content"]
        # Surface error for debugging
        return None
    except Exception:
        return None


def _generate_anthropic(
    prompt: str, model: str, api_key: Optional[str],
    temperature: float, max_tokens: int,
) -> Optional[str]:
    """Anthropic /v1/messages endpoint."""
    if not api_key:
        return None
    try:
        r = requests.post(
            f"{ANTHROPIC_BASE}/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        if r.ok:
            return r.json()["content"][0]["text"]
    except Exception:
        pass
    return None


# ── Convenience helpers (used by the rest of the app) ────────────────────────

def parse_json_response(response: str) -> Optional[dict]:
    """Strip markdown fences and parse first JSON object from a response."""
    if not response:
        return None
    clean = re.sub(r'```(?:json)?', '', response).strip()
    clean = re.sub(r'```', '', clean).strip()
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def get_registry() -> list[dict]:
    return MODEL_REGISTRY


def get_registry_by_group() -> dict[str, list[dict]]:
    """Return models grouped by their group key."""
    groups: dict[str, list] = {}
    for m in MODEL_REGISTRY:
        groups.setdefault(m["group"], []).append(m)
    return groups


def has_credential(model_key: str, session_overrides: dict = None) -> bool:
    """Return True if the model has the credential it needs."""
    entry = _REGISTRY_MAP.get(model_key)
    if not entry:
        return False
    if entry["group"] == "ollama":
        return True   # no credential needed
    return bool(_resolve_api_key(entry, session_overrides))


def model_label(model_key: str) -> str:
    entry = _REGISTRY_MAP.get(model_key, {})
    return entry.get("label", model_key)
