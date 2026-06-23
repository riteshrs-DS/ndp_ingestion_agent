"""
eval_page.py  ·  Streamlit UI for the System Evaluation tab
─────────────────────────────────────────────────────────────
Import and call render_eval_tab() from app.py.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import streamlit as st
import pandas as pd
from datetime import datetime

from eval.eval_engine import (
    run_all_fixtures, summarise, build_token_summary,
    simulate_llm_token_scenarios, GROUND_TRUTH_FIXTURES,
    ExtractionResult, estimate_tokens,
)
from utils.llm_registry import MODEL_REGISTRY, GROUP_META


# ── Colour helpers ────────────────────────────────────────────────────────────
def _pct_color(pct: float) -> str:
    if pct >= 90: return "#27ae60"
    if pct >= 70: return "#f39c12"
    return "#e74c3c"


def _bar(pct: float, color: str = "#1a5276", height: int = 8) -> str:
    return (
        f'<div style="background:var(--color-background-secondary);border-radius:4px;height:{height}px;width:100%">'
        f'<div style="background:{color};width:{min(pct,100):.1f}%;height:{height}px;border-radius:4px;transition:width .4s"></div>'
        f'</div>'
    )


def _metric(label: str, value: str, color: str = "#1a5276") -> str:
    return (
        f'<div style="background:var(--color-background-secondary);border-radius:8px;'
        f'padding:0.7rem 1rem;text-align:center">'
        f'<div style="font-size:1.6rem;font-weight:600;color:{color}">{value}</div>'
        f'<div style="font-size:0.78rem;color:var(--color-text-secondary)">{label}</div>'
        f'</div>'
    )


# ── Main render function ──────────────────────────────────────────────────────

def render_eval_tab():
    st.markdown("### 📊 System Evaluation")
    st.markdown(
        "Measures extraction **accuracy**, **timing**, and **token consumption** "
        "across all loaders and LLM models using ground-truth fixture datasets."
    )

    # ── Run / cache controls ──────────────────────────────────────────────────
    col_run, col_clear = st.columns([2, 1])
    with col_run:
        run_btn = st.button("▶ Run Full Evaluation", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑 Clear Results", use_container_width=True):
            for k in ["_eval_results", "_eval_summary", "_eval_token_rows", "_eval_ts"]:
                st.session_state.pop(k, None)
            st.rerun()

    if run_btn:
        progress_bar = st.progress(0.0, text="Starting evaluation…")
        log_ph = st.empty()
        log_lines = []

        def _cb(msg, pct):
            log_lines.append(msg)
            log_ph.markdown("\n".join(f"- {m}" for m in log_lines[-6:]))
            progress_bar.progress(min(pct, 1.0), text=msg)

        t_start = time.perf_counter()
        results = run_all_fixtures(progress_cb=_cb)
        wall_ms = (time.perf_counter() - t_start) * 1000

        summary = summarise(results)
        model_keys = [m["key"] for m in MODEL_REGISTRY]
        token_rows = build_token_summary(model_keys)

        st.session_state["_eval_results"]    = results
        st.session_state["_eval_summary"]    = summary
        st.session_state["_eval_token_rows"] = token_rows
        st.session_state["_eval_ts"]         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["_eval_wall_ms"]    = wall_ms

        progress_bar.empty()
        log_ph.empty()
        st.rerun()

    results:    list[ExtractionResult] = st.session_state.get("_eval_results", [])
    summary:    dict                   = st.session_state.get("_eval_summary", {})
    token_rows: list[dict]             = st.session_state.get("_eval_token_rows", [])

    if not results:
        st.info("Click **▶ Run Full Evaluation** to start. "
                f"This evaluates {len(GROUND_TRUTH_FIXTURES)} fixtures across TXT, XML, MLCommons, and HuggingFace loaders.")
        return

    ts = st.session_state.get("_eval_ts", "")
    wall = st.session_state.get("_eval_wall_ms", 0)
    st.caption(f"Last run: {ts}  ·  Total wall time: {wall:.0f} ms  ·  {len(results)} fixtures")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Overview KPIs
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎯 Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Completeness", f"{summary['avg_completeness_pct']}%",  _pct_color(summary['avg_completeness_pct'])),
        (c2, "Exact Accuracy", f"{summary['avg_accuracy_pct']}%",    _pct_color(summary['avg_accuracy_pct'])),
        (c3, "Overall Score",  f"{summary['avg_overall_score']}",     _pct_color(summary['avg_overall_score'])),
        (c4, "Preflight Pass", f"{summary['preflight_pass_rate']}%",  _pct_color(summary['preflight_pass_rate'])),
        (c5, "Avg Time",       f"{summary['avg_time_ms']} ms",      "#27ae60"),
    ]
    for col, label, val, color in kpis:
        with col:
            st.markdown(_metric(label, val, color), unsafe_allow_html=True)

    # Per-source breakdown
    st.markdown("**Results by source type:**")
    src_cols = st.columns(len(summary.get("by_source", {})))
    for col, (src, data) in zip(src_cols, summary["by_source"].items()):
        with col:
            st.markdown(
                f'<div style="border:1px solid var(--color-border-tertiary);border-radius:8px;padding:0.6rem">'
                f'<b>{src}</b> ({data["count"]} fixtures)<br>'
                f'Completeness: <b>{data["avg_completeness"]}%</b><br>'
                f'Accuracy: <b>{data["avg_accuracy"]}%</b><br>'
                f'Avg time: <b>{data["avg_time_ms"]} ms</b>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Accuracy Detail
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 1️⃣ Accuracy")
    st.markdown(
        "Scoring per field: **Exact match** = 1.0 · "
        "**Partial match** (≥60% word overlap) = 0.4–0.6 · "
        "**Present but wrong** = 0.1–0.3 · **Missing** = 0.0"
    )

    # Summary table
    acc_rows = []
    for r in results:
        acc_rows.append({
            "Fixture ID":     r.fixture_id,
            "Source":         r.source_type,
            "Loader":         r.loader_name,
            "Fields tested":  r.fields_required,
            "Present":        r.fields_present,
            "Exact match":    r.fields_exact,
            "Partial match":  r.fields_partial,
            "Completeness %": r.completeness_pct,
            "Accuracy %":     r.accuracy_pct,
            "Overall score":  r.overall_score,
            "Preflight":      "✅" if r.preflight_passed else "❌",
            "Format errors":  len(r.format_errors),
        })

    # Color-code the key numeric columns manually (no matplotlib needed)
    df_acc = pd.DataFrame(acc_rows)

    def _color_pct(val):
        try:
            v = float(val)
            if v >= 90: bg = "#27ae60"
            elif v >= 70: bg = "#f39c12"
            else: bg = "#fadbd8"
            return f"background-color: {bg}"
        except Exception:
            return ""

    styled = df_acc.style.map(_color_pct,
        subset=["Completeness %", "Accuracy %", "Overall score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Per-fixture field drill-down
    st.markdown("**Field-level drill-down** — select a fixture:")
    fixture_ids = [r.fixture_id for r in results]
    chosen_id = st.selectbox("Fixture", fixture_ids, label_visibility="collapsed")
    chosen_r = next(r for r in results if r.fixture_id == chosen_id)

    field_rows = []
    for fr in chosen_r.field_results:
        if fr.exact_match:
            status = "✅ Exact"
            color  = "#d5f5e3"
        elif fr.partial_match:
            status = "🟡 Partial"
            color  = "#fef9e7"
        elif fr.present:
            status = "🟠 Present"
            color  = "#fdebd0"
        else:
            status = "❌ Missing"
            color  = "#fadbd8"
        field_rows.append({
            "Field":         fr.field,
            "Status":        status,
            "Score":         fr.score,
            "Expected":      fr.expected[:80],
            "Extracted":     fr.got[:80],
            "Format valid":  "✅" if fr.format_valid else "❌",
        })
    df_fields = pd.DataFrame(field_rows)
    st.dataframe(df_fields, use_container_width=True, hide_index=True)

    if chosen_r.format_errors:
        st.warning("**Format errors detected:**\n" + "\n".join(f"- {e}" for e in chosen_r.format_errors))
    if chosen_r.missing_required:
        st.error("**Missing required fields:**\n" + "\n".join(f"- `{f}`" for f in chosen_r.missing_required))

    # Visual progress bars per fixture
    st.markdown("**Score bars — all fixtures:**")
    for r in results:
        col_a, col_b, col_c = st.columns([3, 1, 1])
        with col_a:
            st.markdown(
                f"**{r.fixture_id}** `{r.source_type}`<br>"
                + _bar(r.completeness_pct, _pct_color(r.completeness_pct), 6),
                unsafe_allow_html=True
            )
        with col_b:
            st.markdown(f"Complete: **{r.completeness_pct}%**")
        with col_c:
            st.markdown(f"Exact: **{r.accuracy_pct}%**")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Timing
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 2️⃣ Extraction Time")
    st.markdown(
        "Wall-clock time for each loader to parse the input and return a flat "
        "metadata dict. Does **not** include LLM call time (depends on model/hardware)."
    )

    timing_rows = []
    for r in results:
        timing_rows.append({
            "Fixture":    r.fixture_id,
            "Source":     r.source_type,
            "Loader":     r.loader_name,
            "Time (ms)":  r.extraction_time_ms,
            "Speed":      "🟢 Fast" if r.extraction_time_ms < 5
                          else ("🟡 Medium" if r.extraction_time_ms < 100 else "🔴 Slow"),
        })

    df_timing = pd.DataFrame(timing_rows)

    def _color_time(val):
        try:
            v = float(val)
            if v < 5:   return "background-color: #27ae60"
            if v < 100: return "background-color: #fef9e7"
            return "background-color: #fadbd8"
        except Exception:
            return ""

    st.dataframe(
        df_timing.style.map(_color_time, subset=["Time (ms)"]),
        use_container_width=True, hide_index=True
    )

    # Bar chart
    st.markdown("**Time per fixture (ms):**")
    chart_data = pd.DataFrame({
        "Fixture": [r.fixture_id for r in results],
        "ms":      [r.extraction_time_ms for r in results],
    }).set_index("Fixture")
    st.bar_chart(chart_data, height=220)

    # Stats
    times = [r.extraction_time_ms for r in results]
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1: st.metric("Min", f"{min(times):.2f} ms")
    with tc2: st.metric("Max", f"{max(times):.2f} ms")
    with tc3: st.metric("Mean", f"{sum(times)/len(times):.2f} ms")
    with tc4: st.metric("Total (8 fixtures)", f"{sum(times):.2f} ms")

    st.markdown("**Note on LLM timing:** Ollama llama3 on a modern laptop averages 3–8 s "
                "for a normalization call. Cloud models (Claude, GPT-4o-mini, ELLM) average "
                "1–4 s depending on network latency. Full pipeline = loader time + LLM time.")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Token Consumption
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 3️⃣ Token Consumption")
    st.markdown(
        "Token counts are estimated using a character-ratio approximation "
        "(cl100k_base: ~4 chars/token for prose, ~3.5 for JSON). "
        "Each extraction cycle = **3 LLM calls**: normalize → repair → questions."
    )

    if not token_rows:
        st.info("Token data not available. Run the evaluation first.")
        return

    # Cost comparison table
    df_tok = pd.DataFrame(token_rows)
    df_tok_disp = df_tok.rename(columns={
        "model_key":         "Model",
        "prompt_tokens":     "Prompt tokens",
        "response_tokens":   "Response tokens",
        "total_tokens":      "Total tokens",
        "cost_per_call_usd": "Cost / cycle (USD)",
        "cost_per_100_usd":  "Cost / 100 cycles",
        "input_rate":        "Input rate",
        "output_rate":       "Output rate",
    })
    def _color_cost(val):
        try:
            v = float(str(val).replace("$",""))
            if v == 0:    return "background-color: #27ae60"
            if v < 0.001: return "background-color: #2980b9"
            if v < 0.003: return "background-color: #fef9e7"
            return "background-color: #fadbd8"
        except Exception:
            return ""

    st.dataframe(
        df_tok_disp.style
            .format({"Cost / cycle (USD)": "${:.6f}", "Cost / 100 cycles": "${:.4f}"})
            .map(_color_cost, subset=["Cost / cycle (USD)"]),
        use_container_width=True, hide_index=True
    )

    # Group breakdown
  #  st.markdown("**Cost comparison by provider group:**")
  #  group_agg = {}
  #  for row in token_rows:
  #      grp = row["model_key"].split("/")[0]
  #      group_agg.setdefault(grp, []).append(row["cost_per_call_usd"])

  #  gc1, gc2, gc3, gc4 = st.columns(4)
  # group_cols = {"ollama": gc1, "ellm": gc2, "anthropic": gc3, "openai": gc4}
   # group_labels = {
    #    "ollama":    "🖥️ Ollama (local)",
    #    "ellm":      "🔬 NRP-ELLM",
    #    "anthropic": "🤖 Anthropic",
    #    "openai":    "🌐 OpenAI",
    # }
    #for grp, col in group_cols.items():
     #   costs = group_agg.get(grp, [0])
     #   avg_cost = sum(costs) / len(costs)
     #   with col:
      #      st.markdown(
       #         _metric(group_labels[grp],
        #                "Free" if avg_cost == 0 else f"${avg_cost:.5f}",
         #               "#27ae60" if avg_cost == 0 else "#1a5276"),
          #      unsafe_allow_html=True
           # )

    # Per-call type breakdown (simulate current active model)
   # st.markdown("**Call-type breakdown — active model:**")
   # active_key = st.session_state.get("active_model_key", "ollama/llama3")
   # call_records = simulate_llm_token_scenarios(active_key)

    #call_df = pd.DataFrame([{
     #   "Call type":       r.call_type,
     #   "Prompt tokens":   r.prompt_tokens,
     #   "Response tokens": r.response_tokens,
     #   "Total tokens":    r.total_tokens,
     #   "Cost (USD)":      r.cost_usd,
      #  "Prompt preview":  r.prompt_text[:80] + "…",
    #} for r in call_records])
    #st.dataframe(call_df, use_container_width=True, hide_index=True)

    #tok_totals = {
    #    "prompt":   sum(r.prompt_tokens for r in call_records),
    #    "response": sum(r.response_tokens for r in call_records),
    #    "total":    sum(r.total_tokens for r in call_records),
    #    "cost":     sum(r.cost_usd for r in call_records),
    #}
    #t1, t2, t3, t4 = st.columns(4)
    #with t1: st.metric("Total prompt tokens",   tok_totals["prompt"])
    #with t2: st.metric("Total response tokens", tok_totals["response"])
    #with t3: st.metric("Total tokens / cycle",  tok_totals["total"])
    #with t4: st.metric("Cost / cycle",
     #                   "Free" if tok_totals["cost"] == 0 else f"${tok_totals['cost']:.5f}") */

    # Projection table
    st.markdown("**Cost projection — how many datasets can I process?**")
    volumes = [10, 50, 100, 500, 1000]
    proj_rows = []
    for key in [r["model_key"] for r in token_rows]:
        rec = next(r for r in token_rows if r["model_key"] == key)
        row = {"Model": key}
        for n in volumes:
            cost = rec["cost_per_call_usd"] * n
            row[f"{n} datasets"] = "Free" if cost == 0 else f"${cost:.4f}"
        proj_rows.append(row)
    df_proj = pd.DataFrame(proj_rows)
    st.dataframe(df_proj, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Export
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📥 Export Results")
    export_data = {
        "generated_at":  st.session_state.get("_eval_ts", ""),
        "summary":       summary,
        "accuracy": [
            {
                "fixture_id":       r.fixture_id,
                "source_type":      r.source_type,
                "loader_name":      r.loader_name,
                "completeness_pct": r.completeness_pct,
                "accuracy_pct":     r.accuracy_pct,
                "overall_score":    r.overall_score,
                "time_ms":          r.extraction_time_ms,
                "preflight_passed": r.preflight_passed,
                "missing_required": r.missing_required,
                "format_errors":    r.format_errors,
                "field_results": [
                    {
                        "field":        fr.field,
                        "expected":     fr.expected,
                        "got":          fr.got,
                        "score":        fr.score,
                        "exact_match":  fr.exact_match,
                        "partial_match":fr.partial_match,
                        "format_valid": fr.format_valid,
                    }
                    for fr in r.field_results
                ],
            }
            for r in results
        ],
        "tokens": token_rows,
    }

    col_j, col_c = st.columns(2)
    with col_j:
        st.download_button(
            "⬇️ Download JSON report",
            data=json.dumps(export_data, indent=2),
            file_name=f"ndp_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_c:
        csv_rows = []
        for r in results:
            for fr in r.field_results:
                csv_rows.append({
                    "fixture_id": r.fixture_id,
                    "source_type": r.source_type,
                    "field": fr.field,
                    "score": fr.score,
                    "exact_match": fr.exact_match,
                    "partial_match": fr.partial_match,
                    "expected": fr.expected,
                    "got": fr.got,
                })
        st.download_button(
            "⬇️ Download CSV (field-level)",
            data=pd.DataFrame(csv_rows).to_csv(index=False),
            file_name=f"ndp_eval_fields_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
