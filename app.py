import os
import streamlit as st
import src.ckan_client as cmod
st.write("ckan_client file:", cmod.__file__)
st.write("CkanClient has list_org_packages:", hasattr(cmod.CkanClient, "list_org_packages"))


from src.config import settings
from src.llm_ollama import OllamaClient
from src.mcp_tools import (
    tool_save_bytes, tool_extract_facts, tool_generate_ckan_json,
    tool_validate_metadata, tool_save_json
)
from src.ckan_client import CkanClient

st.set_page_config(page_title="Multi-Source Metadata → JSON", layout="wide")
st.title("Multi-Source Metadata → CKAN JSON (PDF / TXT / XML / EML)")

with st.sidebar:
    st.header("LLM (Ollama)")
    model = st.text_input("Ollama model", value=settings.ollama_model)
    base_url = st.text_input("Ollama base URL", value=settings.ollama_base_url)
    max_pages = st.number_input("Max PDF pages", 1, 20, settings.max_pdf_pages)

    st.header("Output")
    out_dir = st.text_input("Output dir", value=settings.output_dir)

tab_upload, tab_ckan = st.tabs(["Upload file", "Pull examples from NDP CKAN"])

# --- Upload tab ---
with tab_upload:
    input_type = st.selectbox("Input type", ["pdf", "txt", "xml", "eml"])
    up = st.file_uploader("Upload metadata file", type=[input_type])

    if up and st.button("Generate JSON from upload"):
        llm = OllamaClient(base_url=base_url, model=model)
        saved = tool_save_bytes(up.getvalue(), out_dir, up.name)
        facts = tool_extract_facts(saved, input_type=input_type, max_pdf_pages=int(max_pages))

        st.subheader("Extracted facts")
        st.json(facts)

        meta = tool_generate_ckan_json(facts, llm)
        st.subheader("Generated CKAN JSON")
        st.json(meta)

        ok, err = tool_validate_metadata(meta)
        st.success("✅ Valid" if ok else "⚠️ Invalid")
        if err: st.code(err)

        base_name = os.path.splitext(up.name)[0] + "_ckan"
        json_path = tool_save_json(meta, out_dir, base_name)
        with open(json_path, "rb") as f:
            st.download_button("Download JSON", f.read(), file_name=os.path.basename(json_path), mime="application/json")

# --- CKAN pull tab ---
with tab_ckan:
    st.caption("Fetch example metadata resources from NDP test catalog (CKAN).")

    ckan_base = st.text_input("CKAN base URL", value="https://ndp-test.sdsc.edu/catalog")
    org = st.selectbox("Organization", ["bco_weather", "clm_test", "wfsi"])
    rows = st.number_input("How many datasets to list", 1, 50, 10)

    if st.button("List datasets"):
        ckan = CkanClient(ckan_base)
        res = ckan.list_org_packages(org, rows=int(rows), start=0)
        st.session_state["datasets"] = res.get("results", [])

    datasets = st.session_state.get("datasets", [])
    if datasets:
        names = [d.get("name") or d.get("id") for d in datasets]
        chosen = st.selectbox("Pick a dataset", names)

        if st.button("Fetch first resource + generate JSON"):
            ckan = CkanClient(ckan_base)
            pkg = ckan.get_package(chosen)
         #   pkg = ckan.package_show(chosen)
            resources = pkg.get("resources", [])
            if not resources:
                st.error("No resources found for this dataset.")
                st.stop()

            r0 = resources[0]
            r_url = r0.get("url")
            r_format = (r0.get("format") or "").lower()

            #content = ckan.download_resource(r_url)
            content = ckan.download(r_url)

            filename = f"{chosen}.{r_format or 'txt'}"
            saved = tool_save_bytes(content, out_dir, filename)

            # Map format -> loader input_type
            if r_format in ("xml",):
                input_type = "xml"
            elif r_format in ("eml",):
                input_type = "eml"
            else:
                input_type = "txt"

            llm = OllamaClient(base_url=base_url, model=model)
            facts = tool_extract_facts(saved, input_type=input_type, max_pdf_pages=int(max_pages))
            facts["ckan_dataset_stub"] = {"name": pkg.get("name"), "title": pkg.get("title")}
            facts["ckan_resource_stub"] = {"url": r_url, "format": r_format, "name": r0.get("name")}

            st.subheader("Extracted facts")
            st.json(facts)

            meta = tool_generate_ckan_json(facts, llm)
            st.subheader("Generated CKAN JSON")
            st.json(meta)
