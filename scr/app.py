"""
NDP Intelligent Data Ingestion Agent
Main Streamlit Application
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import re
import time
import requests
import streamlit as st
from datetime import datetime

from utils.schema import (
    PUBLIC_REQUIRED_FIELDS, get_missing_required_fields,
    get_empty_ckan_package, flatten_ckan_package
)
from utils.validator import (
    preflight_check, validate_with_ndp_api, build_ckan_package,
    format_validation_report, parse_validation_errors
)
from utils.llm_provider import (
    check_ollama_status, normalize_metadata_with_llm,
    repair_metadata_with_llm, generate_questions_for_missing_fields,
    DEFAULT_MODEL, DEFAULT_BASE_URL,
)
from utils.llm_registry import (
    get_registry_by_group, check_model_connectivity,
    has_credential, model_label, GROUP_META,
    DEFAULT_MODEL_KEY, MODEL_REGISTRY,
)
from loaders.xml_loader import (
    parse_iso_xml, parse_eml_xml,
    fetch_ckan_packages, normalize_from_ckan_api,
    deep_extract_from_ckan_package
)
from loaders.mlcommons_loader import (
    get_mlcommons_catalog, extract_mlcommons_dataset, normalize_mlcommons_to_flat
)
from loaders.huggingface_loader import (
    check_hf_connectivity, search_hf_datasets, extract_hf_dataset,
    repo_id_from_url, get_popular_datasets_catalog, popular_dataset_to_flat,
    HF_DATASETS_URL,
)
from eval.eval_page import render_eval_tab
from loaders.text_loader import parse_txt_metadata, clean_text
from loaders.pdf_loader import parse_pdf_metadata

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NDP Ingestion Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def _build_session_overrides() -> dict:
    """
    Collect all runtime credentials + config from st.session_state
    into the dict format expected by llm_registry functions.
    """
    overrides = {}
    # Ollama URL
    if st.session_state.get("OLLAMA_BASE_URL"):
        overrides["OLLAMA_BASE_URL"] = st.session_state["OLLAMA_BASE_URL"]
    # API keys stored under cred_<ENV_VAR_NAME>
    for key, val in st.session_state.items():
        if key.startswith("cred_") and val:
            env_name = key[len("cred_"):]
            overrides[env_name] = val
    return overrides

# ─── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main color palette */
    :root {
        --ndp-blue: #1a5276;
        --ndp-teal: #148f77;
        --ndp-gold: #d4ac0d;
        --ndp-light: #eaf2ff;
        --ndp-red: #c0392b;
    }
    .main-header {
        background: linear-gradient(135deg, #1a5276, #148f77);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #cde8ff; margin: 4px 0 0 0; font-size: 0.95rem; }

    .status-card {
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
    }
    #.status-success { background: #d5f5e3; border-color: #27ae60; }
    #.status-error { background: #fadbd8; border-color: #e74c3c; }
    #.status-warning { background: #fef9e7; border-color: #f39c12; }
    #.status-info { background: #eaf2ff; border-color: #2980b9; }
    
    
    .status-success { background: #006633; border-color: #27ae60; }
    .status-error { background: #FF0000; border-color: #e74c3c; }
    .status-warning { background: #FF8000; border-color: #f39c12; }
    .status-info { background: #eaf2ff; border-color: #2980b9; }

    .field-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }
    .field-required { border-left: 3px solid #e74c3c; }
    .field-populated { border-left: 3px solid #27ae60; }
    .field-missing { border-left: 3px solid #f39c12; }

    .section-header {
        background: #1a5276;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin: 1rem 0 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .step-badge {
        display: inline-block;
        background: #148f77;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        text-align: center;
        line-height: 28px;
        font-weight: bold;
        margin-right: 8px;
    }
    .json-block {
        background: #1e1e2e;
        color: #cdd6f4;
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.82rem;
        overflow-x: auto;
        max-height: 400px;
        overflow-y: auto;
    }
    .metric-box {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-box .metric-val { font-size: 2rem; font-weight: bold; color: #1a5276; }
    .metric-box .metric-label { font-size: 0.8rem; color: #666; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
    }
    .ndp-pill {
        display: inline-block;
        background: #eaf2ff;
        border: 1px solid #2980b9;
        color: #1a5276;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "flat_metadata": {},
        "ckan_package": {},
        "validation_result": None,
        "preflight_result": None,
        "api_result": None,
        "missing_fields": [],
        "user_answers": {},
        "workflow_step": 1,
        "active_model_key": DEFAULT_MODEL_KEY,
        "OLLAMA_BASE_URL": DEFAULT_BASE_URL,
        "source_label": "",
        "history": [],
        "is_private": False,
        "owner_org": "ndp",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Agent Configuration")

    # ── LLM Provider Panel ────────────────────────────────────────────────────
    with st.expander("🤖 LLM Provider & Model", expanded=True):

        # Build grouped options for the selectbox
        groups    = get_registry_by_group()
        options   = []   # list of model_key strings
        labels    = []   # display labels with group prefix
        for group_key in ["ollama", "ellm", "anthropic", "openai"]:
            group_models = groups.get(group_key, [])
            for m in group_models:
                options.append(m["key"])
                labels.append(f"{GROUP_META[group_key]['label']}  ·  {m['label']}")

        saved_key = st.session_state.get("active_model_key", DEFAULT_MODEL_KEY)
        saved_idx = options.index(saved_key) if saved_key in options else 0

        chosen_idx = st.selectbox(
            "Active model",
            range(len(options)),
            format_func=lambda i: labels[i],
            index=saved_idx,
            help="All models use the same NDP normalization prompts."
        )
        chosen_key = options[chosen_idx]
        st.session_state["active_model_key"] = chosen_key

        chosen_entry = next(m for m in MODEL_REGISTRY if m["key"] == chosen_key)
        st.caption(f"ℹ️ {chosen_entry['notes']}")

        # ── Credentials ───────────────────────────────────────────────────────
        group = chosen_entry["group"]
        env_name = chosen_entry.get("api_key_env")

        if group == "ollama":
            ollama_url = st.text_input(
                "Ollama base URL",
                value=st.session_state.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
                key="ollama_url_input"
            )
            st.session_state["OLLAMA_BASE_URL"] = ollama_url

        elif env_name:
            # Check if already set via .env
            import os
            env_val = os.environ.get(env_name, "")
            placeholder = "Loaded from .env ✓" if env_val else f"Paste {env_name} here…"
            runtime_key = st.text_input(
                f"API Key ({env_name})",
                type="password",
                placeholder=placeholder,
                value=st.session_state.get(f"cred_{env_name}", ""),
                key=f"cred_input_{env_name}",
                help=f"Set once — shared by all {group.upper()} models."
            )
            if runtime_key:
                st.session_state[f"cred_{env_name}"] = runtime_key

        # ── Test connection ────────────────────────────────────────────────────
        if st.button("🔍 Test Connection", use_container_width=True):
            overrides = _build_session_overrides()
            ok, msg = check_model_connectivity(chosen_key, session_overrides=overrides)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        # ── Credential status badges ──────────────────────────────────────────
        st.markdown("**All model credential status:**")
        for group_key, group_models in groups.items():
            st.markdown(f"*{GROUP_META[group_key]['label']}*")
            for m in group_models:
                overrides = _build_session_overrides()
                cred_ok = has_credential(m["key"], session_overrides=overrides)
                is_active = m["key"] == chosen_key
                icon = "🟢" if cred_ok else "🔴"
                active_tag = " ◀ active" if is_active else ""
                st.markdown(f"  {icon} `{m['key']}`{active_tag}")

    # ── Dataset settings ──────────────────────────────────────────────────────
    with st.expander("📋 Dataset Settings"):
        is_private = st.checkbox("Private Dataset", value=st.session_state.is_private)
        st.session_state.is_private = is_private
        owner_org = st.text_input("Owner Organization", value=st.session_state.owner_org)
        st.session_state.owner_org = owner_org

    # ── NDP Settings ──────────────────────────────────────────────────────────
    with st.expander("🌐 NDP Settings"):
        ndp_base = st.text_input("NDP CKAN Base URL", value="https://ndp-test.sdsc.edu/catalog")
        validate_url = st.text_input("Validate Endpoint",
                                      value="https://ndp-test.sdsc.edu/catalog2/ndp/package_validate")

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    flat = st.session_state.flat_metadata
    total_fields = len(PUBLIC_REQUIRED_FIELDS)
    filled = sum(1 for f in PUBLIC_REQUIRED_FIELDS
                 if flat.get(f) and (not isinstance(flat[f], list) or len(flat[f]) > 0))
    pct = int((filled / total_fields) * 100) if total_fields else 0

    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-val">{pct}%</div>
        <div class="metric-label">Fields Populated ({filled}/{total_fields})</div>
    </div>
    """, unsafe_allow_html=True)

    active_label = model_label(st.session_state.get("active_model_key", DEFAULT_MODEL_KEY))
    st.markdown(f"**Model:** `{active_label}`")
    if st.session_state.flat_metadata:
        st.markdown(f"**Source:** `{st.session_state.source_label}`")

    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
        for k in ["flat_metadata", "ckan_package", "validation_result", "preflight_result",
                  "api_result", "missing_fields", "user_answers", "workflow_step",
                  "source_label", "history"]:
            st.session_state[k] = {} if k in ["flat_metadata", "ckan_package", "user_answers"] else \
                                   [] if k in ["missing_fields", "history"] else \
                                   1 if k == "workflow_step" else None
        st.rerun()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 An Intelligent Data Ingestion Agent</h1>
    <p>Automated metadata normalization & validation for the National Data Platform (NDP) </p>
</div>
""", unsafe_allow_html=True)

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab_ingest, tab_review, tab_validate, tab_submit, tab_explore, tab_eval = st.tabs([
    "📥 1. Ingest Metadata",
    "✏️ 2. Review & Enrich",
    "✅ 3. Validate",
    "🚀 4. Submit",
    "🔭 5. Explore NDP",
    "📊 6. Evaluation",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: INGEST METADATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.markdown("### <span class='step-badge'>1</span> Choose Your Metadata Source", unsafe_allow_html=True)
    st.markdown("Select a source type and provide the input. The agent will extract and normalize metadata automatically.")

    source_type = st.radio(
        "Source Type",
        ["📄 Upload File (PDF/XML/TXT)", "🌐 NDP CKAN Catalog URL",
         "🤖 MLCommons Datasets", "🤗 HuggingFace Datasets",
         "✍️ Paste Raw Text", "🔗 Custom URL"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # ── Source: File Upload ──────────────────────────────────────────────────
    if source_type == "📄 Upload File (PDF/XML/TXT)":
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded = st.file_uploader(
                "Upload a metadata file",
                type=["pdf", "xml", "txt", "eml"],
                help="Supported: PDF documents, ISO/XML, EML, or plain-text metadata files"
            )
        with col2:
            st.markdown("""
            **Supported Formats:**
            - 📑 PDF documents (reports, data summaries)
            - 🗂️ ISO 19139 XML metadata
            - 🌿 EML (Ecological Metadata Language)
            - 📝 Plain text descriptions
            """)

        if uploaded:
            st.info(f"📎 File loaded: **{uploaded.name}** ({uploaded.size:,} bytes)")
            if st.button("🤖 Extract Metadata", type="primary", use_container_width=False):
                with st.spinner("Parsing file and extracting metadata..."):
                    file_bytes = uploaded.read()
                    filename = uploaded.name.lower()

                    if filename.endswith('.pdf'):
                        result = parse_pdf_metadata(file_bytes, filename=uploaded.name)
                    elif filename.endswith('.xml') or filename.endswith('.eml'):
                        try:
                            text = file_bytes.decode('utf-8', errors='replace')
                        except Exception:
                            text = file_bytes.decode('latin-1', errors='replace')
                        # Detect EML vs ISO
                        if 'eml://ecoinformatics' in text or '<eml:eml' in text.lower():
                            result = parse_eml_xml(text)
                        else:
                            result = parse_iso_xml(text)
                    elif filename.endswith('.txt'):
                        try:
                            text = file_bytes.decode('utf-8', errors='replace')
                        except Exception:
                            text = file_bytes.decode('latin-1', errors='replace')
                        result = parse_txt_metadata(clean_text(text))
                    else:
                        st.error("Unsupported file type.")
                        result = {}

                    if result.get('error'):
                        st.error(f"Parse error: {result['error']}")
                    else:
                        st.session_state.flat_metadata = result
                        st.session_state.source_label = f"File: {uploaded.name}"
                        st.session_state.workflow_step = 2
                        st.success(f"✅ Extracted {len([v for v in result.values() if v and not str(v).startswith('_')])} metadata fields!")
                        st.rerun()

    # ── Source: NDP CKAN Catalog ─────────────────────────────────────────────
    elif source_type == "🌐 NDP CKAN Catalog URL":

        st.markdown(
            "Fetch metadata records from the NDP test CKAN catalog. "
            "The agent will **deep-extract** metadata from the actual ISO XML / EML / TXT "
            "source behind each package, not just the top-level CKAN fields."
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            ckan_url = st.text_input("CKAN Base URL", value="https://ndp-test.sdsc.edu/catalog")
            org_options = {
                "bco_weather (170 ISO/XML records)": "bco_weather",
                "clm_test (188 TXT records)": "clm_test",
                "wfsi (100 EML records)": "wfsi",
                "Custom organization": "__custom__"
            }
            org_choice = st.selectbox("Organization", list(org_options.keys()))
            org = org_options[org_choice]
            if org == "__custom__":
                org = st.text_input("Enter organization name:")
            max_rows = st.slider("Max packages to fetch", 1, 20, 5)

        with col2:
            st.markdown("""
            **Available Organizations:**
            | Org | Format | Count |
            |-----|--------|-------|
            | bco_weather | ISO/XML | 170 |
            | clm_test | TXT | 188 |
            | wfsi | EML | 100 |

            **Deep-Extraction Layers:**
            1. `package_show` (full extras)
            2. ISO XML endpoint
            3. Resource XML/EML/TXT URLs
            4. Notes-text heuristics
            """)

        fetch_clicked = st.button("📡 Fetch Packages from NDP", type="primary")
        if fetch_clicked:
            with st.spinner(f"Fetching package list from `{org}`…"):
                packages = fetch_ckan_packages(ckan_url, org, max_rows=max_rows)
            if packages and packages[0].get("error"):
                st.error(f"Error: {packages[0]['error']}")
                st.info("The NDP catalog may require network access from your environment.")
            elif not packages:
                st.warning("No packages found for this organization.")
            else:
                st.session_state["_ckan_packages"] = packages
                st.session_state["_ckan_url"] = ckan_url
                st.session_state["_ckan_org"] = org
                st.rerun()

        # Show package list if already fetched
        packages = st.session_state.get("_ckan_packages", [])
        if packages and not packages[0].get("error"):
            st.success(f"✅ Found {len(packages)} package(s) from `{st.session_state.get('_ckan_org', org)}`.")

            pkg_labels = [
                f"{p.get('title', p.get('name', f'Package {i}'))}"
                for i, p in enumerate(packages)
            ]
            selected_idx = st.selectbox(
                "Select a package to extract metadata from:",
                range(len(pkg_labels)),
                format_func=lambda i: pkg_labels[i]
            )
            selected = packages[selected_idx]

            # Show raw CKAN preview
            with st.expander("📄 Raw CKAN Package (top-level fields)", expanded=False):
                st.json({
                    k: v for k, v in selected.items()
                    if k not in ("resources", "extras", "tags", "relationships_as_object",
                                 "relationships_as_subject", "organization")
                })
                if selected.get("extras"):
                    st.markdown("**extras:**")
                    st.json(selected["extras"])
                if selected.get("resources"):
                    st.markdown("**resources:**")
                    st.json(selected["resources"])

            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                deep_btn = st.button(
                    "🔬 Deep Extract Metadata",
                    type="primary",
                    help="Fetches ISO XML / EML / TXT source and merges all layers",
                    use_container_width=True
                )
            with col_b:
                basic_btn = st.button(
                    "⚡ Quick Ingest (CKAN fields only)",
                    help="Uses only the top-level CKAN API fields — faster but less complete",
                    use_container_width=True
                )

            if deep_btn:
                log_container = st.empty()
                progress_log = []

                def update_log(msg):
                    progress_log.append(msg)
                    log_container.markdown(
                        "\n".join(f"- {m}" for m in progress_log[-6:])
                    )

                with st.spinner("Running deep metadata extraction…"):
                    result = deep_extract_from_ckan_package(
                        selected,
                        base_url=st.session_state.get("_ckan_url", ckan_url),
                        org=st.session_state.get("_ckan_org", org),
                        progress_cb=update_log
                    )

                log_container.empty()

                if result:
                    st.session_state.flat_metadata = result
                    st.session_state.source_label = (
                        f"NDP CKAN (deep): {org}/{selected.get('name')} "
                        f"[{result.get('source_format', '')}]"
                    )
                    st.session_state.workflow_step = 2

                    filled = sum(1 for v in result.values()
                                 if v and not str(v).startswith('_'))
                    src = result.get('source_format', 'CKAN API')
                    deep_src = result.get('_deep_source_url', '')

                    st.success(f"✅ Deep extraction complete — **{filled} fields** extracted | Source: `{src}`")
                    if deep_src:
                        st.info(f"🔗 Deep source: `{deep_src}`")

                    # Show what was extracted vs what the basic CKAN had
                    basic = normalize_from_ckan_api(selected)
                    new_fields = [
                        k for k, v in result.items()
                        if v and not k.startswith('_')
                        and (not basic.get(k) or basic.get(k) != v)
                        and k != 'source_format'
                    ]
                    if new_fields:
                        with st.expander(f"🆕 {len(new_fields)} field(s) gained vs basic CKAN"):
                            for f in new_fields:
                                st.markdown(f"**`{f}`** → `{str(result[f])[:120]}`")
                    st.rerun()
                else:
                    st.error("Extraction returned no data.")

            if basic_btn:
                result = normalize_from_ckan_api(selected)
                st.session_state.flat_metadata = result
                st.session_state.source_label = f"NDP CKAN: {org}/{selected.get('name')}"
                st.session_state.workflow_step = 2
                st.success("✅ Basic CKAN fields loaded.")
                st.rerun()

    # ── Source: MLCommons Datasets ───────────────────────────────────────────
    elif source_type == "🤖 MLCommons Datasets":
        st.markdown(
            "Browse and ingest metadata from the **[MLCommons dataset catalog]"
            "(https://mlcommons.org/datasets/)**. "
            "Metadata is extracted from the live MLCommons website and normalized "
            "into NDP/CKAN-compatible fields."
        )

        catalog = get_mlcommons_catalog()

        # Dataset cards grid
        st.markdown("#### 📦 Available MLCommons Datasets")
        col1, col2 = st.columns(2)
        for i, ds in enumerate(catalog):
            with (col1 if i % 2 == 0 else col2):
                with st.container(border=True):
                    st.markdown(f"##### {ds['title']}")
                    st.markdown(f"_{ds['summary']}_")
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.markdown(f"**Format:** {ds.get('format','—')}")
                    with meta_cols[1]:
                        st.markdown(f"**License:** {ds.get('license','—')}")
                    with meta_cols[2]:
                        st.markdown(f"**Size:** {ds.get('size','—')}")
                    if ds.get('tags'):
                        st.markdown(" ".join(f"`{t}`" for t in ds['tags'][:5]))

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button(f"🔬 Live Extract", key=f"mlc_live_{ds['slug']}",
                                     help="Fetch latest data from mlcommons.org"):
                            prog_log = []
                            prog_ph = st.empty()
                            def _log(m, ph=prog_ph, log=prog_log):
                                log.append(m)
                                ph.markdown("\n".join(f"- {x}" for x in log[-5:]))
                            with st.spinner(f"Extracting {ds['title']}…"):
                                raw = extract_mlcommons_dataset(
                                    ds['slug'], live_scrape=True, progress_cb=_log
                                )
                            prog_ph.empty()
                            flat = normalize_mlcommons_to_flat(raw)
                            st.session_state.flat_metadata = flat
                            st.session_state.source_label = f"MLCommons: {ds['title']} [{flat.get('source_format','')}]"
                            st.session_state.workflow_step = 2
                            filled = sum(1 for v in flat.values() if v and not str(v).startswith('_'))
                            st.success(f"✅ {filled} fields extracted!")
                            st.rerun()
                    with btn_col2:
                        if st.button(f"⚡ Catalog", key=f"mlc_cat_{ds['slug']}",
                                     help="Use pre-loaded catalog data (no network needed)"):
                            raw = extract_mlcommons_dataset(
                                ds['slug'], live_scrape=False
                            )
                            flat = normalize_mlcommons_to_flat(raw)
                            st.session_state.flat_metadata = flat
                            st.session_state.source_label = f"MLCommons: {ds['title']} (catalog)"
                            st.session_state.workflow_step = 2
                            st.success(f"✅ Loaded from catalog!")
                            st.rerun()

        st.markdown("---")
        st.markdown("##### 🔗 Enter a Custom MLCommons Dataset URL")
        custom_mlc = st.text_input(
            "MLCommons dataset page URL:",
            placeholder="https://mlcommons.org/datasets/peoples-speech/"
        )
        if st.button("🔬 Extract from URL", type="secondary") and custom_mlc:
            slug_match = re.search(r'/datasets/([^/]+)/?$', custom_mlc)
            slug = slug_match.group(1) if slug_match else custom_mlc.split('/')[-2]
            with st.spinner(f"Fetching {custom_mlc}…"):
                raw = extract_mlcommons_dataset(slug, live_scrape=True)
            if raw.get('error'):
                st.error(raw['error'])
            else:
                flat = normalize_mlcommons_to_flat(raw)
                st.session_state.flat_metadata = flat
                st.session_state.source_label = f"MLCommons URL: {custom_mlc}"
                st.session_state.workflow_step = 2
                st.success(f"✅ Extracted {len([v for v in flat.values() if v])} fields!")
                st.rerun()

    # ── Source: HuggingFace Datasets ─────────────────────────────────────────
    elif source_type == "🤗 HuggingFace Datasets":
        st.markdown(
            "Search and ingest metadata from the **[HuggingFace Hub]"
            "(https://huggingface.co/datasets)** — 100,000+ public datasets. "
            "Uses the HF Hub REST API. An optional API token unlocks gated datasets."
        )

        # ── HF token + connectivity ───────────────────────────────────────────
        with st.expander("🔑 HuggingFace API Token (optional — needed for gated datasets)", expanded=False):
            hf_token = st.text_input("HF Token", type="password",
                                     value=st.session_state.get("hf_token", ""),
                                     placeholder="hf_xxxxxxxxxxxxxxxxxxxx")
            if hf_token:
                st.session_state["hf_token"] = hf_token
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                if st.button("🔍 Test Connection", key="hf_test_conn"):
                    ok, msg = check_hf_connectivity(token=hf_token or None)
                    if ok:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")
            with col_t2:
                st.markdown("[Get a free token →](https://huggingface.co/settings/tokens)")

        hf_token_val = st.session_state.get("hf_token", "") or None

        # ── Mode selector ─────────────────────────────────────────────────────
        hf_mode = st.radio(
            "How to find a dataset:",
            ["🔎 Search HuggingFace", "🔗 Enter dataset ID or URL", "⭐ Browse popular datasets"],
            horizontal=True, label_visibility="visible"
        )

        # ── Mode: Search ──────────────────────────────────────────────────────
        if hf_mode == "🔎 Search HuggingFace":
            col_s1, col_s2, col_s3 = st.columns([3, 1, 1])
            with col_s1:
                hf_query = st.text_input("Search datasets", placeholder="e.g. climate, speech, medical imaging")
            with col_s2:
                hf_sort  = st.selectbox("Sort by", ["downloads", "likes", "lastModified", "trending"])
            with col_s3:
                hf_limit = st.selectbox("Max results", [5, 10, 20, 50], index=1)

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                hf_author = st.text_input("Filter by author/org", placeholder="e.g. google, facebook")
            with col_f2:
                hf_tag    = st.text_input("Filter by task tag", placeholder="e.g. text-classification")

            if st.button("🔍 Search HuggingFace", type="primary"):
                with st.spinner(f"Searching HuggingFace for '{hf_query}'…"):
                    results = search_hf_datasets(
                        query=hf_query, author=hf_author,
                        tags=[hf_tag] if hf_tag else None,
                        limit=hf_limit, sort=hf_sort, token=hf_token_val
                    )
                if not results:
                    st.warning("No results found or HuggingFace API not reachable. "
                               "Try **Browse popular datasets** for offline access.")
                else:
                    st.session_state["_hf_search_results"] = results
                    st.rerun()

            results = st.session_state.get("_hf_search_results", [])
            if results:
                st.success(f"✅ {len(results)} dataset(s) found")
                for ds in results:
                    gated_badge = " 🔒" if ds.get("gated") else ""
                    with st.expander(
                        f"**{ds['pretty_name'] or ds['id']}**{gated_badge}  —  "
                        f"⬇️ {ds['downloads']:,}  ❤️ {ds['likes']:,}  📐 {ds['size_label']}"
                    ):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**ID:** `{ds['id']}`")
                            st.markdown(f"**License:** {ds['license'] or '—'}")
                            st.markdown(f"**Tasks:** {', '.join(ds['tasks'][:4]) or '—'}")
                        with c2:
                            st.markdown(f"**Languages:** {', '.join(ds['languages'][:4]) or '—'}")
                            st.markdown(f"**Last modified:** {ds['lastModified'][:10] if ds['lastModified'] else '—'}")
                            st.markdown(f"[🔗 View on HuggingFace]({ds['url']})")
                        if ds.get("tags"):
                            st.markdown(" ".join(f"`{t}`" for t in ds["tags"][:8]))

                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("🔬 Full Extract", key=f"hf_extract_{ds['id'].replace('/','_')}",
                                         help="Fetch complete metadata via Hub API + README"):
                                log_ph = st.empty()
                                log_items = []
                                def _hf_log(m, ph=log_ph, li=log_items):
                                    li.append(m); ph.markdown("\n".join(f"- {x}" for x in li[-5:]))
                                with st.spinner(f"Extracting {ds['id']}…"):
                                    flat = extract_hf_dataset(ds["id"], token=hf_token_val, progress_cb=_hf_log)
                                log_ph.empty()
                                st.session_state.flat_metadata = flat
                                st.session_state.source_label = f"HuggingFace: {ds['id']}"
                                st.session_state.workflow_step = 2
                                st.success(f"✅ {sum(1 for v in flat.values() if v)} fields extracted!")
                                st.rerun()
                        with col_b:
                            if st.button("⚡ Quick Load", key=f"hf_quick_{ds['id'].replace('/','_')}",
                                         help="Load from search result fields only"):
                                from loaders.huggingface_loader import popular_dataset_to_flat
                                flat = popular_dataset_to_flat(ds)
                                st.session_state.flat_metadata = flat
                                st.session_state.source_label = f"HuggingFace: {ds['id']} (quick)"
                                st.session_state.workflow_step = 2
                                st.success("✅ Quick-loaded!")
                                st.rerun()

        # ── Mode: ID / URL ────────────────────────────────────────────────────
        elif hf_mode == "🔗 Enter dataset ID or URL":
            st.markdown("Enter a HuggingFace dataset repo ID (`owner/name`) or the full URL:")
            hf_input = st.text_input(
                "Dataset ID or URL",
                placeholder="e.g.  nyu-mll/glue  or  https://huggingface.co/datasets/squad"
            )

            col_x1, col_x2 = st.columns(2)
            with col_x1:
                do_full   = st.button("🔬 Full Extract", type="primary")
            with col_x2:
                do_readme = st.button("📄 README only")

            if hf_input and (do_full or do_readme):
                # Resolve repo_id
                if hf_input.startswith("http"):
                    repo_id = repo_id_from_url(hf_input)
                    if not repo_id:
                        st.error("Could not parse repo ID from that URL.")
                        repo_id = None
                else:
                    repo_id = hf_input.strip().strip("/")

                if repo_id:
                    log_ph2 = st.empty()
                    log2 = []
                    def _hf_log2(m, ph=log_ph2, li=log2):
                        li.append(m); ph.markdown("\n".join(f"- {x}" for x in li[-5:]))

                    with st.spinner(f"Extracting `{repo_id}`…"):
                        flat = extract_hf_dataset(repo_id, token=hf_token_val, progress_cb=_hf_log2)
                    log_ph2.empty()

                    if flat.get("title"):
                        st.session_state.flat_metadata = flat
                        st.session_state.source_label = f"HuggingFace: {repo_id}"
                        st.session_state.workflow_step = 2
                        filled = sum(1 for v in flat.values() if v and not str(v).startswith("_"))
                        src = flat.get("source_format", "HuggingFace Hub")
                        st.success(f"✅ {filled} fields extracted from `{repo_id}` [{src}]")
                        if flat.get("_source_url"):
                            st.info(f"🔗 {flat['_source_url']}")
                        st.rerun()
                    else:
                        st.error(
                            f"Could not extract metadata for `{repo_id}`. "
                            "Check the repo ID or try with an API token."
                        )

        # ── Mode: Popular / Curated ───────────────────────────────────────────
        elif hf_mode == "⭐ Browse popular datasets":
            st.markdown("Curated popular HuggingFace datasets — **works offline**, no API call needed.")
            popular = get_popular_datasets_catalog()
            col_p1, col_p2 = st.columns(2)
            for i, ds in enumerate(popular):
                with (col_p1 if i % 2 == 0 else col_p2):
                    with st.container(border=True):
                        st.markdown(f"##### {ds.get('pretty_name', ds['id'])}")
                        st.markdown(f"_{ds['description'][:160]}…_")
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1: st.markdown(f"**License:** `{ds.get('license','—')}`")
                        with mc2: st.markdown(f"**Size:** {ds.get('size_label','—')}")
                        with mc3: st.markdown(f"⬇️ {ds.get('downloads',0):,}")
                        if ds.get("tasks"):
                            st.markdown(" ".join(f"`{t}`" for t in ds["tasks"][:4]))
                        st.markdown(f"[🔗 {ds['id']}]({HF_DATASETS_URL}/{ds['id']})")

                        bc1, bc2 = st.columns(2)
                        with bc1:
                            if st.button("🔬 Full Extract", key=f"pop_full_{ds['id'].replace('/','_')}"):
                                log_p = st.empty(); log_pi = []
                                def _pl(m, ph=log_p, li=log_pi):
                                    li.append(m); ph.markdown("\n".join(f"- {x}" for x in li[-4:]))
                                with st.spinner(f"Extracting {ds['id']}…"):
                                    flat = extract_hf_dataset(ds["id"], token=hf_token_val, progress_cb=_pl)
                                log_p.empty()
                                st.session_state.flat_metadata = flat
                                st.session_state.source_label = f"HuggingFace: {ds['id']}"
                                st.session_state.workflow_step = 2
                                st.success("✅ Extracted!")
                                st.rerun()
                        with bc2:
                            if st.button("⚡ Offline Load", key=f"pop_off_{ds['id'].replace('/','_')}"):
                                flat = popular_dataset_to_flat(ds)
                                st.session_state.flat_metadata = flat
                                st.session_state.source_label = f"HuggingFace: {ds['id']} (catalog)"
                                st.session_state.workflow_step = 2
                                st.success("✅ Loaded from catalog!")
                                st.rerun()

    # ── Source: Paste Raw Text ───────────────────────────────────────────────
    elif source_type == "✍️ Paste Raw Text":
        fmt = st.radio("Text Format", ["Plain Text / TXT", "ISO/XML", "EML/XML"], horizontal=True)
        pasted = st.text_area(
            "Paste your metadata content here:",
            height=300,
            placeholder="Paste ISO XML, EML XML, or plain-text metadata description..."
        )
        if st.button("🤖 Parse Metadata", type="primary") and pasted:
            with st.spinner("Parsing metadata..."):
                if fmt == "Plain Text / TXT":
                    result = parse_txt_metadata(clean_text(pasted))
                elif fmt == "ISO/XML":
                    result = parse_iso_xml(pasted)
                else:
                    result = parse_eml_xml(pasted)

            if result.get("error"):
                st.error(f"Parse error: {result['error']}")
            else:
                st.session_state.flat_metadata = result
                st.session_state.source_label = f"Pasted {fmt}"
                st.session_state.workflow_step = 2
                st.success(f"✅ Parsed successfully!")
                st.rerun()

    # ── Source: Custom URL ───────────────────────────────────────────────────
    elif source_type == "🔗 Custom URL":
        custom_url = st.text_input(
            "Enter URL to fetch metadata from:",
            placeholder="https://example.com/dataset/metadata.xml"
        )
        url_fmt = st.radio("Expected Format", ["Auto-detect", "ISO/XML", "EML/XML", "Plain Text"], horizontal=True)

        if st.button("🌐 Fetch & Parse", type="primary") and custom_url:
            with st.spinner("Fetching URL..."):
                try:
                    resp = requests.get(custom_url, timeout=20)
                    resp.raise_for_status()
                    content = resp.text

                    if url_fmt == "Auto-detect":
                        ct = resp.headers.get('Content-Type', '')
                        if 'xml' in ct or content.strip().startswith('<'):
                            if 'eml' in content.lower()[:200]:
                                result = parse_eml_xml(content)
                            else:
                                result = parse_iso_xml(content)
                        else:
                            result = parse_txt_metadata(clean_text(content))
                    elif url_fmt == "ISO/XML":
                        result = parse_iso_xml(content)
                    elif url_fmt == "EML/XML":
                        result = parse_eml_xml(content)
                    else:
                        result = parse_txt_metadata(clean_text(content))

                    st.session_state.flat_metadata = result
                    st.session_state.source_label = f"URL: {custom_url[:60]}"
                    st.session_state.workflow_step = 2
                    st.success("✅ Fetched and parsed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to fetch URL: {e}")

    # ── Current extracted metadata summary ──────────────────────────────────
    if st.session_state.flat_metadata:
        st.markdown("---")
        st.markdown("### 📋 Currently Loaded Metadata")
        flat = st.session_state.flat_metadata
        pfr = preflight_check(flat, is_private=st.session_state.is_private)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Fields Populated", sum(1 for v in flat.values() if v and not str(v).startswith('_')))
        with c2:
            st.metric("Missing Required", len(pfr["missing_fields"]))
        with c3:
            st.metric("Source Format", flat.get('source_format', 'Unknown'))

        if pfr["passed"]:
            st.markdown('<div class="status-card status-success">✅ All required fields are populated! Proceed to Review.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-card status-warning">⚠️ {len(pfr["missing_fields"])} required field(s) missing. Use the Review tab to fill them in.</div>', unsafe_allow_html=True)

        st.markdown("➡️ **Next:** Go to the **Review & Enrich** tab to inspect, edit, and use AI to fill missing fields.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: REVIEW & ENRICH
# ══════════════════════════════════════════════════════════════════════════════
with tab_review:
    st.markdown("### <span class='step-badge'>2</span> Review & Enrich Metadata", unsafe_allow_html=True)

    if not st.session_state.flat_metadata:
        st.info("👈 Please ingest metadata first using the **Ingest Metadata** tab.")
    else:
        flat = st.session_state.flat_metadata
        pfr = preflight_check(flat, is_private=st.session_state.is_private)
        missing = pfr["missing_fields"]
        st.session_state.missing_fields = missing

        # ── AI Enrichment ────────────────────────────────────────────────────
        if missing:
            st.markdown("#### 🤖 AI-Assisted Enrichment")
            st.markdown(f"**{len(missing)} required field(s) are missing.** Choose how to fill them:")

            enrich_col1, enrich_col2 = st.columns(2)
            with enrich_col1:
                if st.button("✨ Auto-fill with LLM", type="primary", help="Use the active LLM model to infer missing fields from existing data"):
                    active_key = st.session_state.get("active_model_key", DEFAULT_MODEL_KEY)
                    overrides  = _build_session_overrides()
                    with st.spinner(f"🤖 Asking `{model_label(active_key)}` to infer missing fields…"):
                        llm_result = normalize_metadata_with_llm(
                            flat, missing,
                            model_key=active_key,
                            session_overrides=overrides,
                        )
                    if llm_result:
                        flat.update(llm_result)
                        st.session_state.flat_metadata = flat
                        st.success(f"✅ LLM filled {len(llm_result)} field(s)!")
                        st.rerun()
                    else:
                        st.error("❌ LLM did not return results. Check provider connection in the sidebar.")
            with enrich_col2:
                st.markdown("*or fill manually in the form below*")

        # ── Manual Edit Form ─────────────────────────────────────────────────
        st.markdown("#### ✏️ Edit Metadata Fields")

        with st.form("metadata_edit_form"):
            # General Section
            st.markdown('<div class="section-header">📁 General</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Title *", value=flat.get('title', ''),
                                       help="Short, descriptive title of the dataset")
                upload_type = st.selectbox("Type of Entity *",
                                            ["dataset", "service", "model", "collection"],
                                            index=["dataset", "service", "model", "collection"].index(
                                                flat.get('extras.uploadType', 'dataset')
                                                if flat.get('extras.uploadType', 'dataset') in ["dataset", "service", "model", "collection"]
                                                else 'dataset'
                                            ))
            with col2:
                data_type = st.text_input("Format of Data",
                                           value=flat.get('extras.dataType', ''),
                                           placeholder="e.g., tabular, timeseries, imagery, text")

            notes = st.text_area("Description *",
                                  value=flat.get('notes', ''),
                                  height=120,
                                  help="Detailed description of the dataset contents and scope")

            tags_raw = flat.get('tags', [])
            if isinstance(tags_raw, list):
                tags_str = ", ".join(tags_raw)
            else:
                tags_str = str(tags_raw)
            tags_input = st.text_input("Keywords / Tags * (comma-separated)",
                                        value=tags_str,
                                        help="At least 1 keyword required for public datasets")

            # Contributors
            st.markdown('<div class="section-header">👤 Contributors & Contact</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                poc_name = st.text_input("Point of Contact Name *",
                                          value=flat.get('extras.pocName', ''))
            with col2:
                poc_email = st.text_input("Point of Contact Email *",
                                           value=flat.get('extras.pocEmail', ''))

            # General Metadata
            st.markdown('<div class="section-header">📅 General Metadata</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                issue_date = st.text_input("Date of Creation *",
                                            value=flat.get('extras.issueDate', ''),
                                            placeholder="YYYY-MM-DD")
            with col2:
                update_date = st.text_input("Date of Last Update *",
                                             value=flat.get('extras.lastUpdateDate', ''),
                                             placeholder="YYYY-MM-DD")

            # Resource
            st.markdown('<div class="section-header">📦 Resource</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                res_name = st.text_input("Resource Title *",
                                          value=flat.get('resource.name', ''))
                res_mimetype = st.text_input("MIME Type *",
                                              value=flat.get('resource.mimetype', ''),
                                              placeholder="e.g., text/csv, application/json")
                res_url = st.text_input("Resource URL",
                                         value=flat.get('resource.url', ''),
                                         placeholder="https://...")
            with col2:
                res_desc = st.text_area("Resource Description *",
                                         value=flat.get('resource.description', ''),
                                         height=80)
                res_format = st.text_input("Format",
                                            value=flat.get('resource.format', ''),
                                            placeholder="e.g., CSV, JSON, XML, NetCDF")
                res_status = st.selectbox("Status",
                                           ["active", "archived", "deprecated"],
                                           index=["active", "archived", "deprecated"].index(
                                               flat.get('resource.status', 'active')
                                               if flat.get('resource.status', 'active') in ["active", "archived", "deprecated"]
                                               else 'active'
                                           ))

            submitted = st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True)

        if submitted:
            # Update flat metadata
            flat['title'] = title
            flat['notes'] = notes
            flat['tags'] = [t.strip() for t in tags_input.split(',') if t.strip()]
            flat['extras.uploadType'] = upload_type
            flat['extras.dataType'] = data_type
            flat['extras.pocName'] = poc_name
            flat['extras.pocEmail'] = poc_email
            flat['extras.issueDate'] = issue_date
            flat['extras.lastUpdateDate'] = update_date
            flat['resource.name'] = res_name
            flat['resource.description'] = res_desc
            flat['resource.mimetype'] = res_mimetype
            flat['resource.format'] = res_format
            flat['resource.status'] = res_status
            flat['resource.url'] = res_url
            st.session_state.flat_metadata = flat
            st.success("✅ Metadata saved!")
            st.rerun()

        # ── Missing Fields Q&A ───────────────────────────────────────────────
        pfr2 = preflight_check(st.session_state.flat_metadata, is_private=st.session_state.is_private)
        remaining_missing = pfr2["missing_fields"]
        if remaining_missing:
            st.markdown("---")
            st.markdown("#### ❓ Agent Questions – Please Fill Missing Fields")
            st.markdown("The following required fields are still missing. Please answer:")

            questions = generate_questions_for_missing_fields(remaining_missing)
            with st.form("agent_questions_form"):
                answers = {}
                for field, question in questions.items():
                    if field == "resource.status":
                        answers[field] = st.selectbox(question, ["active", "archived", "deprecated"])
                    elif field == "extras.uploadType":
                        answers[field] = st.selectbox(question, ["dataset", "service", "model", "collection"])
                    elif "date" in field.lower():
                        answers[field] = st.text_input(question, placeholder="YYYY-MM-DD",
                                                        value=st.session_state.user_answers.get(field, ''))
                    elif field == "notes" or "description" in field.lower():
                        answers[field] = st.text_area(question,
                                                       value=st.session_state.user_answers.get(field, ''))
                    elif field == "tags":
                        answers[field] = st.text_input(question, placeholder="keyword1, keyword2, ...",
                                                        value=st.session_state.user_answers.get(field, ''))
                    else:
                        answers[field] = st.text_input(question,
                                                        value=st.session_state.user_answers.get(field, ''))

                apply_answers = st.form_submit_button("✅ Apply Answers", type="primary")

            if apply_answers:
                flat = st.session_state.flat_metadata
                for field, answer in answers.items():
                    if answer and str(answer).strip():
                        if field == 'tags':
                            flat[field] = [t.strip() for t in answer.split(',') if t.strip()]
                        else:
                            flat[field] = answer
                st.session_state.flat_metadata = flat
                st.session_state.user_answers.update(answers)
                st.success("✅ Answers applied!")
                st.rerun()
        else:
            st.markdown('<div class="status-card status-success">✅ All required fields are populated! Ready for validation.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: VALIDATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_validate:
    st.markdown("### <span class='step-badge'>3</span> Validate Metadata", unsafe_allow_html=True)

    if not st.session_state.flat_metadata:
        st.info("👈 Please ingest and review metadata first.")
    else:
        flat = st.session_state.flat_metadata
        ckan_pkg = build_ckan_package(flat,
                                       is_private=st.session_state.is_private,
                                       owner_org=st.session_state.owner_org)
        st.session_state.ckan_package = ckan_pkg

        col1, col2 = st.columns([1.5, 1])

        with col1:
            # ── Preflight ─────────────────────────────────────────────────────
            st.markdown("#### 🔍 Preflight Check (Local)")
            pfr = preflight_check(flat, is_private=st.session_state.is_private)

            if pfr["passed"]:
                st.markdown('<div class="status-card status-success">✅ Preflight PASSED — no missing required fields detected.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-card status-error">❌ Preflight FAILED</div>', unsafe_allow_html=True)
                if pfr["missing_fields"]:
                    st.markdown("**Missing required fields:**")
                    for f in pfr["missing_fields"]:
                        st.markdown(f"  - `{f}`: {PUBLIC_REQUIRED_FIELDS.get(f, {}).get('description', '')}")

            if pfr["errors"]:
                for e in pfr["errors"]:
                    st.error(e)
            if pfr["warnings"]:
                for w in pfr["warnings"]:
                    st.warning(w)

            # ── NDP API Validation ────────────────────────────────────────────
            st.markdown("#### 🌐 NDP API Validation")
            st.markdown("Submit to the NDP validation endpoint to confirm CKAN compatibility.")

            if not pfr["passed"]:
                st.warning("⚠️ Preflight failed — fix missing fields before running API validation.")

            if st.button("🚀 Run NDP Validation", type="primary", disabled=False):
                with st.spinner("Calling NDP validation endpoint..."):
                    api_result = validate_with_ndp_api(ckan_pkg)
                    st.session_state.api_result = api_result

                    # Log to history
                    st.session_state.history.append({
                        "timestamp": datetime.now().isoformat(),
                        "attempt": len(st.session_state.history) + 1,
                        "passed": api_result.get("passed"),
                        "status_code": api_result.get("status_code"),
                        "errors": parse_validation_errors(api_result)
                    })

            if st.session_state.api_result:
                api = st.session_state.api_result
                if api.get("passed"):
                    st.markdown('<div class="status-card status-success">✅ NDP Validation PASSED!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-card status-error">❌ NDP Validation FAILED</div>', unsafe_allow_html=True)
                    errors = parse_validation_errors(api)
                    if errors:
                        st.markdown("**Errors:**")
                        for e in errors:
                            st.error(e)

                    # LLM Repair
                    if errors and st.button("🔧 Auto-Repair with LLM", type="secondary"):
                        active_key = st.session_state.get("active_model_key", DEFAULT_MODEL_KEY)
                        overrides  = _build_session_overrides()
                        with st.spinner(f"🔧 Asking `{model_label(active_key)}` to repair the metadata JSON…"):
                            repaired = repair_metadata_with_llm(
                                ckan_pkg, errors,
                                st.session_state.user_answers,
                                model_key=active_key,
                                session_overrides=overrides,
                            )
                        if repaired:
                            # Re-flatten repaired package
                            new_flat = flatten_ckan_package(repaired)
                            new_flat['source_format'] = flat.get('source_format', 'unknown')
                            new_flat['_raw_text'] = flat.get('_raw_text', '')
                            st.session_state.flat_metadata = new_flat
                            st.session_state.ckan_package = repaired
                            st.session_state.api_result = None
                            st.success("✅ Repaired! Re-run validation to check.")
                            st.rerun()
                        else:
                            st.error("LLM repair failed. Check Ollama connection.")

            # ── Validation History ────────────────────────────────────────────
            if st.session_state.history:
                st.markdown("#### 📈 Validation Attempts")
                for h in st.session_state.history:
                    icon = "✅" if h["passed"] else "❌"
                    st.markdown(f"{icon} **Attempt {h['attempt']}** — {h['timestamp'][:19]} | HTTP {h['status_code']}")

        with col2:
            # ── CKAN JSON Preview ─────────────────────────────────────────────
            st.markdown("#### 📄 Generated CKAN JSON")
            json_str = json.dumps(ckan_pkg, indent=2)
            st.markdown(f'<div class="json-block"><pre>{json_str}</pre></div>', unsafe_allow_html=True)

            # Download
            st.download_button(
                "⬇️ Download CKAN JSON",
                data=json_str,
                file_name=f"ckan_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

            # Field coverage heatmap
            st.markdown("#### 📊 Field Coverage")
            for field, info in PUBLIC_REQUIRED_FIELDS.items():
                val = flat.get(field)
                is_filled = bool(val and (not isinstance(val, list) or len(val) > 0))
                icon = "🟢" if is_filled else "🔴"
                short_val = str(val)[:40] + "..." if val and len(str(val)) > 40 else str(val) if val else "—"
                st.markdown(f"{icon} **{field}** `{short_val}`")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SUBMIT
# ══════════════════════════════════════════════════════════════════════════════
with tab_submit:
    st.markdown("### <span class='step-badge'>4</span> Submit to NDP Catalog", unsafe_allow_html=True)

    if not st.session_state.ckan_package:
        st.info("👈 Please complete validation first.")
    else:
        ckan_pkg = st.session_state.ckan_package
        api_result = st.session_state.api_result

        if not api_result or not api_result.get("passed"):
            st.warning("⚠️ Metadata has not passed NDP validation. We recommend validating first before submission.")

        st.markdown("#### 📋 Submission Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Title:** {ckan_pkg.get('title', '—')}")
            st.markdown(f"**Organization:** {ckan_pkg.get('owner_org', '—')}")
            st.markdown(f"**Visibility:** {'🔒 Private' if ckan_pkg.get('private') else '🌐 Public'}")
        with col2:
            tags = [t.get('name', t) for t in ckan_pkg.get('tags', [])]
            st.markdown(f"**Tags:** {', '.join(tags) or '—'}")
            extras = {e['key']: e['value'] for e in ckan_pkg.get('extras', [])}
            st.markdown(f"**Type:** {extras.get('uploadType', '—')}")
            resources = ckan_pkg.get('resources', [{}])
            st.markdown(f"**Resources:** {len(resources)} resource(s)")

        st.markdown("---")
        st.markdown("#### 🚀 Submission Options")
        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:
            st.markdown("**Option 1: Download for Manual Submission**")
            st.markdown("Download the validated CKAN JSON and submit via the NDP portal or CLI.")
            json_str = json.dumps(ckan_pkg, indent=2)
            st.download_button(
                "⬇️ Download CKAN JSON",
                data=json_str,
                file_name=f"ndp_{ckan_pkg.get('name', 'dataset')}.json",
                mime="application/json",
                use_container_width=True
            )

            # Sample curl command
            with st.expander("📟 Sample curl command"):
                curl = f"""curl -X POST https://ndp-test.sdsc.edu/catalog2/ndp/package_validate \\
-H "Content-Type: application/json" \\
-d '{json.dumps(ckan_pkg)}'"""
                st.code(curl, language="bash")

        with sub_col2:
            st.markdown("**Option 2: Submit via CKAN API**")
            st.markdown("Provide your API key to submit directly. *(Requires CKAN write access)*")

            api_key = st.text_input("CKAN API Key", type="password",
                                     placeholder="Your CKAN API key...")
            ckan_submit_url = st.text_input("CKAN Submit URL",
                                             value="https://ndp-test.sdsc.edu/catalog/api/3/action/package_create")

            if st.button("🚀 Submit to NDP", type="primary", disabled=not api_key):
                with st.spinner("Submitting to NDP CKAN..."):
                    try:
                        resp = requests.post(
                            ckan_submit_url,
                            json=ckan_pkg,
                            headers={
                                "Content-Type": "application/json",
                                "X-CKAN-API-Key": api_key
                            },
                            timeout=30
                        )
                        if resp.status_code in (200, 201):
                            result = resp.json()
                            if result.get("success"):
                                st.success("🎉 Dataset successfully submitted to NDP!")
                                pkg_id = result.get("result", {}).get("id", "")
                                if pkg_id:
                                    st.markdown(f"**Package ID:** `{pkg_id}`")
                            else:
                                st.error(f"Submission failed: {result.get('error', 'Unknown error')}")
                        else:
                            st.error(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    except Exception as e:
                        st.error(f"Submission error: {e}")

        # ── Workflow Summary ─────────────────────────────────────────────────
        if st.session_state.history:
            st.markdown("---")
            st.markdown("#### 📈 Workflow Summary")
            attempts = len(st.session_state.history)
            passed = sum(1 for h in st.session_state.history if h["passed"])
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Attempts", attempts)
            with c2:
                st.metric("Successful", passed)
            with c3:
                st.metric("Source Format", st.session_state.flat_metadata.get('source_format', '—'))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: EXPLORE NDP
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("### <span class='step-badge'>5</span> Explore NDP Data Sources", unsafe_allow_html=True)
    st.markdown("Browse available datasets from the NDP test CKAN catalog and wildfire treatment data.")

    exp_tab1, exp_tab2, exp_tab3, exp_tab4, exp_tab5 = st.tabs([
        "🌊 bco_weather (ISO)", "🌿 wfsi (EML)", "🔥 Wildfire PDF", "🤖 MLCommons", "🤗 HuggingFace"
    ])

    with exp_tab1:
        st.markdown("**BCO Weather Station Datasets** — 170 ISO/XML metadata records")
        if st.button("🔍 Browse bco_weather", key="browse_bco"):
            with st.spinner("Fetching..."):
                pkgs = fetch_ckan_packages("https://ndp-test.sdsc.edu/catalog", "bco_weather", max_rows=10)
            if pkgs and not pkgs[0].get("error"):
                for p in pkgs:
                    with st.expander(f"📦 {p.get('title', p.get('name', 'Unknown'))}"):
                        st.markdown(f"**Name:** {p.get('name', '—')}")
                        st.markdown(f"**Notes:** {(p.get('notes', '') or '')[:300]}...")
                        tags = [t.get('name', '') for t in p.get('tags', [])]
                        st.markdown(f"**Tags:** {', '.join(tags) or '—'}")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button(f"🔬 Deep Extract", key=f"deep_{p.get('name')}"):
                                with st.spinner("Deep extracting…"):
                                    result = deep_extract_from_ckan_package(
                                        p, base_url="https://ndp-test.sdsc.edu/catalog", org="bco_weather"
                                    )
                                st.session_state.flat_metadata = result
                                st.session_state.source_label = f"NDP: bco_weather/{p.get('name')} [{result.get('source_format','')}]"
                                st.success("Loaded! Go to Review tab.")
                        with col_b:
                            if st.button(f"⚡ Basic", key=f"basic_{p.get('name')}"):
                                result = normalize_from_ckan_api(p)
                                st.session_state.flat_metadata = result
                                st.session_state.source_label = f"NDP: bco_weather/{p.get('name')}"
                                st.success("Loaded!")
            elif pkgs and pkgs[0].get("error"):
                st.error(pkgs[0]["error"])

    with exp_tab2:
        st.markdown("**WFSI Datasets** — 100 EML metadata records")
        if st.button("🔍 Browse wfsi", key="browse_wfsi"):
            with st.spinner("Fetching..."):
                pkgs = fetch_ckan_packages("https://ndp-test.sdsc.edu/catalog", "wfsi", max_rows=10)
            if pkgs and not pkgs[0].get("error"):
                for p in pkgs:
                    with st.expander(f"📦 {p.get('title', p.get('name', 'Unknown'))}"):
                        st.markdown(f"**Notes:** {(p.get('notes', '') or '')[:300]}...")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button(f"🔬 Deep Extract", key=f"deep_wfsi_{p.get('name')}"):
                                with st.spinner("Deep extracting…"):
                                    result = deep_extract_from_ckan_package(
                                        p, base_url="https://ndp-test.sdsc.edu/catalog", org="wfsi"
                                    )
                                st.session_state.flat_metadata = result
                                st.session_state.source_label = f"NDP: wfsi/{p.get('name')} [{result.get('source_format','')}]"
                                st.success("Loaded!")
                        with col_b:
                            if st.button(f"⚡ Basic", key=f"basic_wfsi_{p.get('name')}"):
                                result = normalize_from_ckan_api(p)
                                st.session_state.flat_metadata = result
                                st.session_state.source_label = f"NDP: wfsi/{p.get('name')}"
                                st.success("Loaded!")
            elif pkgs and pkgs[0].get("error"):
                st.error(pkgs[0]["error"])

    with exp_tab3:
        st.markdown("**Wildfire Treatment Dashboard PDFs**")
        st.markdown("Upload PDFs from [wildfiretaskforce.org/treatment-dashboard](https://wildfiretaskforce.org/treatment-dashboard/) to extract dataset metadata.")

        wf_pdf = st.file_uploader("Upload wildfire PDF", type=["pdf"], key="wf_pdf")
        if wf_pdf:
            st.info(f"📎 {wf_pdf.name} ({wf_pdf.size:,} bytes)")
            max_p = st.slider("Max pages to parse", 1, 20, 5, key="wf_pages")
            if st.button("🤖 Extract from Wildfire PDF", type="primary"):
                with st.spinner("Extracting metadata from PDF..."):
                    result = parse_pdf_metadata(wf_pdf.read(), filename=wf_pdf.name, max_pages=max_p)
                if result:
                    st.session_state.flat_metadata = result
                    st.session_state.source_label = f"Wildfire PDF: {wf_pdf.name}"
                    st.success(f"✅ Extracted {len([v for v in result.values() if v])} fields!")
                    st.markdown("Go to the **Review & Enrich** tab to complete and validate.")
                    with st.expander("Preview Extracted Fields"):
                        for k, v in result.items():
                            if not k.startswith('_'):
                                st.markdown(f"**{k}:** {str(v)[:200]}")

    with exp_tab4:
        st.markdown("**MLCommons Datasets** — `mlcommons.org/datasets/`")
        st.markdown(
            "MLCommons builds open, large-scale, and diverse datasets for AI evaluation. "
            "Select any dataset below to extract and normalize its metadata for NDP ingestion."
        )

        catalog = get_mlcommons_catalog()

        for ds in catalog:
            with st.expander(f"📦 {ds['title']} — {ds.get('format','?')} · {ds.get('size','?')}"):
                st.markdown(f"_{ds['summary']}_")
                st.markdown(f"**License:** {ds.get('license','—')} &nbsp;|&nbsp; "
                            f"**Tags:** {', '.join(ds.get('tags', [])[:6])}")
                st.markdown(f"🔗 [View on MLCommons]({ds['url']})")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔬 Live Extract", key=f"exp_mlc_live_{ds['slug']}",
                                 help="Fetch latest metadata from mlcommons.org"):
                        prog_log2 = []
                        prog_ph2 = st.empty()
                        def _log2(m, ph=prog_ph2, log=prog_log2):
                            log.append(m)
                            ph.markdown("\n".join(f"- {x}" for x in log[-5:]))
                        with st.spinner(f"Extracting {ds['title']}…"):
                            raw = extract_mlcommons_dataset(
                                ds['slug'], live_scrape=True, progress_cb=_log2
                            )
                        prog_ph2.empty()
                        flat = normalize_mlcommons_to_flat(raw)
                        st.session_state.flat_metadata = flat
                        st.session_state.source_label = (
                            f"MLCommons: {ds['title']} [{flat.get('source_format','')}]"
                        )
                        filled = sum(1 for v in flat.values() if v and not str(v).startswith('_'))
                        st.success(f"✅ {filled} fields loaded — go to Review & Enrich tab.")
                with col_b:
                    if st.button("⚡ Catalog (offline)", key=f"exp_mlc_cat_{ds['slug']}"):
                        raw = extract_mlcommons_dataset(ds['slug'], live_scrape=False)
                        flat = normalize_mlcommons_to_flat(raw)
                        st.session_state.flat_metadata = flat
                        st.session_state.source_label = f"MLCommons: {ds['title']} (catalog)"
                        st.success("✅ Loaded from catalog — go to Review & Enrich tab.")

        st.markdown("---")
        st.markdown("**💡 About MLCommons Datasets**")
        st.markdown("""
        MLCommons produces rigorously validated, open datasets for AI benchmarking and research.
        All datasets are accessible under open licenses (CC-BY or CC-BY-SA 4.0).
        The agent scrapes the live MLCommons website for the latest metadata and maps it
        directly to NDP/CKAN required fields.
        """)
        st.markdown(
            "📎 [Browse all datasets at mlcommons.org/datasets/](https://mlcommons.org/datasets/)"
        )

    with exp_tab5:
        st.markdown("**HuggingFace Hub Datasets** — `huggingface.co/datasets`")
        st.markdown(
            "100,000+ open datasets for ML research. "
            "Search live via the HF API or pick from the curated popular list below."
        )

        hf_token_exp = st.session_state.get("hf_token", "") or None

        # Quick search bar
        col_es1, col_es2 = st.columns([4, 1])
        with col_es1:
            exp_query = st.text_input("Search HuggingFace datasets", key="exp_hf_q",
                                      placeholder="e.g. climate, medical, code")
        with col_es2:
            exp_search_btn = st.button("🔍 Search", key="exp_hf_search")

        if exp_search_btn and exp_query:
            with st.spinner("Searching…"):
                exp_results = search_hf_datasets(
                    query=exp_query, limit=10, sort="downloads", token=hf_token_exp
                )
            if exp_results:
                st.session_state["_exp_hf_results"] = exp_results
            else:
                st.warning("No results or API unreachable — browse popular datasets below.")

        exp_results = st.session_state.get("_exp_hf_results", [])
        if exp_results:
            st.success(f"Found {len(exp_results)} result(s) for your search:")
            for ds in exp_results:
                with st.expander(f"📦 {ds['pretty_name'] or ds['id']}  —  ⬇️ {ds['downloads']:,}"):
                    st.markdown(f"`{ds['id']}` | License: {ds['license'] or '—'} | "
                                f"Size: {ds['size_label'] or '—'}")
                    st.markdown(f"Tasks: {', '.join(ds['tasks'][:4]) or '—'}")
                    st.markdown(f"[🔗 View on HuggingFace]({ds['url']})")
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button("🔬 Full Extract", key=f"exp5_full_{ds['id'].replace('/','_')}"):
                            with st.spinner(f"Extracting {ds['id']}…"):
                                flat = extract_hf_dataset(ds["id"], token=hf_token_exp)
                            st.session_state.flat_metadata = flat
                            st.session_state.source_label = f"HuggingFace: {ds['id']}"
                            st.success("✅ Loaded — go to Review & Enrich tab.")
                    with bc2:
                        if st.button("⚡ Quick Load", key=f"exp5_quick_{ds['id'].replace('/','_')}"):
                            flat = popular_dataset_to_flat(ds)
                            st.session_state.flat_metadata = flat
                            st.session_state.source_label = f"HuggingFace: {ds['id']} (quick)"
                            st.success("✅ Loaded!")

        st.markdown("---")
        st.markdown("#### ⭐ Popular Datasets (offline)")
        popular = get_popular_datasets_catalog()
        for ds in popular:
            with st.expander(f"📦 {ds.get('pretty_name', ds['id'])}  —  {ds.get('size_label','?')}  |  "
                             f"License: {ds.get('license','?')}  |  ⬇️ {ds.get('downloads',0):,}"):
                st.markdown(f"_{ds['description']}_")
                st.markdown(f"**Tasks:** {', '.join(ds.get('tasks',[])[:4])}")
                st.markdown(f"**Languages:** {', '.join(ds.get('languages',[])[:4])}")
                st.markdown(f"[🔗 {ds['id']}]({HF_DATASETS_URL}/{ds['id']})")
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.button("🔬 Full Extract", key=f"exp5p_full_{ds['id'].replace('/','_')}"):
                        with st.spinner(f"Extracting {ds['id']}…"):
                            flat = extract_hf_dataset(ds["id"], token=hf_token_exp)
                        st.session_state.flat_metadata = flat
                        st.session_state.source_label = f"HuggingFace: {ds['id']}"
                        st.success("✅ Loaded — go to Review & Enrich tab.")
                with bc2:
                    if st.button("⚡ Offline Load", key=f"exp5p_off_{ds['id'].replace('/','_')}"):
                        flat = popular_dataset_to_flat(ds)
                        st.session_state.flat_metadata = flat
                        st.session_state.source_label = f"HuggingFace: {ds['id']} (catalog)"
                        st.success("✅ Loaded from catalog!")

    # ── Schema Reference ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📖 NDP/CKAN Required Fields Reference")
    with st.expander("View Full Schema", expanded=False):
        import pandas as pd
        rows = []
        for field, info in PUBLIC_REQUIRED_FIELDS.items():
            rows.append({
                "Field": field,
                "Section": info["section"],
                "CKAN Field": info["ckan_field"],
                "Type": info["data_type"],
                "Description": info["description"][:80] + "..." if len(info["description"]) > 80 else info["description"]
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    render_eval_tab()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.8rem; padding: 1rem 0;">
    🔬 <b>An Intelligent Data Ingestion Agent</b> &nbsp;|&nbsp;
    DSE 260-B MAS in Data Science &amp; Engineering (Cohort 11) &nbsp;|&nbsp;
    Advisors: Ilkay Altintas &amp; Taina Coleman &nbsp;|&nbsp;
    Team: Chung Loh &amp; Ritesh Saxena
</div>
""", unsafe_allow_html=True)
