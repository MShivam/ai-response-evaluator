"""
app.py
------
AI Response Evaluator — Streamlit entry point.

Sections (sidebar navigation):
  1. 🧪 New Evaluation   – submit a prompt, collect responses, score them
  2. 📊 Dashboard         – compare scores with interactive charts
  3. 📜 History           – browse and search past evaluations
  4. 📤 Export            – download CSV / JSON

Run:   streamlit run app.py
"""

import os
import json
import time
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Project modules ────────────────────────────────────────────────────────
from database.db_manager import (
    initialize_db,
    create_evaluation,
    save_response,
    save_score,
    get_responses_for_evaluation,
    get_scores_for_response,
    get_all_evaluations,
    search_evaluations,
    get_full_evaluation_data,
)
from models.ai_clients import get_client, AVAILABLE_MODELS, MOCK_MODEL_NAMES
from evaluators.auto_evaluator import evaluate_response
from utils.chart_utils import (
    build_radar_chart,
    build_bar_chart,
    build_weighted_score_gauge,
    build_history_line_chart,
    build_score_heatmap,
)
from utils.export_utils import (
    export_to_csv,
    export_to_json,
    export_all_to_csv,
    export_all_to_json,
)

# ── App-wide config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Response Evaluator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CRITERIA = ["accuracy", "completeness", "clarity", "creativity", "helpfulness", "overall_quality"]
CRITERIA_LABELS = {
    "accuracy": "Accuracy",
    "completeness": "Completeness",
    "clarity": "Clarity",
    "creativity": "Creativity",
    "helpfulness": "Helpfulness",
    "overall_quality": "Overall Quality",
}

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F1729 0%, #1A2440 100%);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
}
.score-high   { background:#14532d; color:#4ade80; }
.score-mid    { background:#713f12; color:#fbbf24; }
.score-low    { background:#7f1d1d; color:#f87171; }

/* Response card */
.response-card {
    background: #0F1729;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 300px;
    overflow-y: auto;
}

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #E2E8F0;
    border-left: 4px solid #4F8EF7;
    padding-left: 0.75rem;
    margin-bottom: 1rem;
}

/* Rank badge */
.rank-1 { color: #FFD700; font-size: 1.2rem; }
.rank-2 { color: #C0C0C0; font-size: 1.1rem; }
.rank-3 { color: #CD7F32; font-size: 1.0rem; }
</style>
""", unsafe_allow_html=True)


# ── Initialise ─────────────────────────────────────────────────────────────
initialize_db()


# ── Helpers ────────────────────────────────────────────────────────────────

def score_class(score: float) -> str:
    if score >= 7.5:
        return "score-high"
    elif score >= 5.0:
        return "score-mid"
    return "score-low"


def score_badge(score: float) -> str:
    cls = score_class(score)
    return f'<span class="score-badge {cls}">{score:.1f}</span>'


def build_scores_dataframe(evaluation_id: int) -> pd.DataFrame:
    """Flatten all responses + scores for one evaluation into a DataFrame."""
    rows = []
    for resp in get_responses_for_evaluation(evaluation_id):
        for sc in get_scores_for_response(resp["id"]):
            rows.append({
                "Model":         resp["model_name"],
                "Scorer":        sc["scorer_type"].title(),
                "Accuracy":      sc.get("accuracy") or 0,
                "Completeness":  sc.get("completeness") or 0,
                "Clarity":       sc.get("clarity") or 0,
                "Creativity":    sc.get("creativity") or 0,
                "Helpfulness":   sc.get("helpfulness") or 0,
                "Overall":       sc.get("overall_quality") or 0,
                "Weighted ⭐":   sc.get("weighted_score") or 0,
                "Latency (ms)":  resp.get("latency_ms") or 0,
                "Tokens":        resp.get("tokens_used") or 0,
            })
    return pd.DataFrame(rows)


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 AI Evaluator")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🧪 New Evaluation", "📊 Dashboard", "📜 History", "📤 Export"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Optional API key input
    api_key = st.text_input(
        "OpenAI API Key (optional)",
        type="password",
        help="Leave blank to use mock models for demo purposes.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API key set ✓")

    st.markdown("---")
    st.caption("Built with Streamlit · SQLite · Plotly")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — NEW EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

if page == "🧪 New Evaluation":
    st.markdown('<div class="section-header">🧪 New Evaluation</div>', unsafe_allow_html=True)
    st.caption("Submit a prompt to multiple AI models, then score the responses.")

    # ── Step 1: Prompt & model selection ─────────────────────────────────
    with st.container():
        prompt = st.text_area(
            "Your Prompt",
            height=120,
            placeholder="e.g. Explain the concept of transformer neural networks in simple terms.",
        )

        model_options = list(AVAILABLE_MODELS.keys())
        default_mocks = [m for m in model_options if m in MOCK_MODEL_NAMES][:2]

        selected_models = st.multiselect(
            "Select AI Models",
            options=model_options,
            default=default_mocks,
            help="Mock models work without an API key. Real models need OPENAI_API_KEY.",
        )

        col_a, col_b = st.columns([1, 4])
        with col_a:
            run_btn = st.button("▶ Run Evaluation", type="primary", use_container_width=True)

    if run_btn:
        if not prompt.strip():
            st.warning("Please enter a prompt before running.")
            st.stop()
        if not selected_models:
            st.warning("Please select at least one model.")
            st.stop()

        # ── Step 2: Collect responses ─────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📡 Collecting Responses…")

        eval_id = create_evaluation(prompt.strip())
        collected: dict[str, dict] = {}   # model → {resp_text, resp_id, ...}

        progress = st.progress(0)
        status   = st.empty()

        for idx, model_name in enumerate(selected_models):
            status.info(f"Querying **{model_name}**…")
            client = get_client(model_name)
            model_resp = client.generate(prompt.strip())

            resp_id = save_response(
                evaluation_id=eval_id,
                model_name=model_name,
                response_text=model_resp.response_text,
                tokens_used=model_resp.tokens_used,
                latency_ms=model_resp.latency_ms,
            )
            collected[model_name] = {
                "resp_id":   resp_id,
                "resp_text": model_resp.response_text,
                "tokens":    model_resp.tokens_used,
                "latency":   model_resp.latency_ms,
                "is_mock":   model_resp.is_mock,
            }
            progress.progress((idx + 1) / len(selected_models))

        status.success("All responses collected!")

        # ── Display responses side by side ────────────────────────────────
        st.markdown("### 💬 Responses")
        cols = st.columns(len(selected_models))
        for col, model_name in zip(cols, selected_models):
            info = collected[model_name]
            with col:
                badge = "🤖 Mock" if info["is_mock"] else "🌐 Live"
                st.markdown(f"**{model_name}** {badge}")
                st.markdown(
                    f'<div class="response-card">{info["resp_text"]}</div>',
                    unsafe_allow_html=True
                )
                st.caption(
                    f"⏱ {info['latency']:.0f} ms  |  🔤 {info['tokens']} tokens"
                )

        # ── Step 3: Auto-evaluate ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Automated Evaluation")

        auto_results: dict[str, dict] = {}
        for model_name, info in collected.items():
            result = evaluate_response(prompt.strip(), info["resp_text"])
            rd = result.to_dict()
            save_score(
                response_id=info["resp_id"],
                scorer_type="automated",
                accuracy=rd["accuracy"],
                completeness=rd["completeness"],
                clarity=rd["clarity"],
                creativity=rd["creativity"],
                helpfulness=rd["helpfulness"],
                overall_quality=rd["overall_quality"],
                weighted_score=rd["weighted_score"],
                justification=rd["justification"],
            )
            auto_results[model_name] = rd

        # Show auto-score summary table
        auto_rows = []
        for model_name, rd in auto_results.items():
            auto_rows.append({
                "Model":        model_name,
                "Accuracy":     rd["accuracy"],
                "Completeness": rd["completeness"],
                "Clarity":      rd["clarity"],
                "Creativity":   rd["creativity"],
                "Helpfulness":  rd["helpfulness"],
                "Overall":      rd["overall_quality"],
                "Weighted ⭐":  rd["weighted_score"],
            })
        auto_df = pd.DataFrame(auto_rows).set_index("Model")
        st.dataframe(
            auto_df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=10),
            use_container_width=True,
        )

        # Justifications expander
        with st.expander("📝 Automated Justifications"):
            for model_name, rd in auto_results.items():
                st.markdown(f"**{model_name}**")
                for criterion, text in rd["justification"].items():
                    st.markdown(f"- **{CRITERIA_LABELS.get(criterion, criterion)}**: {text}")
                st.markdown("---")

        # ── Step 4: Human scoring ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 👤 Your Manual Scores")
        st.caption("Rate each response on a scale of 1–10 per criterion.")

        with st.form("human_scores_form"):
            human_scores: dict[str, dict[str, float]] = {}

            for model_name in selected_models:
                st.markdown(f"#### {model_name}")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                human_scores[model_name] = {
                    "accuracy":        c1.slider("Accuracy",        1, 10, 7, key=f"acc_{model_name}"),
                    "completeness":    c2.slider("Completeness",    1, 10, 7, key=f"com_{model_name}"),
                    "clarity":         c3.slider("Clarity",         1, 10, 7, key=f"cla_{model_name}"),
                    "creativity":      c4.slider("Creativity",      1, 10, 5, key=f"cre_{model_name}"),
                    "helpfulness":     c5.slider("Helpfulness",     1, 10, 7, key=f"hel_{model_name}"),
                    "overall_quality": c6.slider("Overall Quality", 1, 10, 7, key=f"ovr_{model_name}"),
                }
                st.markdown("---")

            submit_human = st.form_submit_button("💾 Save Human Scores", type="primary")

        if submit_human:
            for model_name, sc in human_scores.items():
                resp_id = collected[model_name]["resp_id"]
                weighted = round(
                    sc["accuracy"] * 0.25 +
                    sc["completeness"] * 0.20 +
                    sc["clarity"] * 0.20 +
                    sc["creativity"] * 0.15 +
                    sc["helpfulness"] * 0.20,
                    2
                )
                save_score(
                    response_id=resp_id,
                    scorer_type="human",
                    accuracy=sc["accuracy"],
                    completeness=sc["completeness"],
                    clarity=sc["clarity"],
                    creativity=sc["creativity"],
                    helpfulness=sc["helpfulness"],
                    overall_quality=sc["overall_quality"],
                    weighted_score=weighted,
                    justification=None,
                )
            st.success("✅ Human scores saved! Check the Dashboard for full analysis.")
            # Store eval_id in session for quick dashboard jump
            st.session_state["last_eval_id"] = eval_id


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.markdown('<div class="section-header">📊 Comparison Dashboard</div>', unsafe_allow_html=True)

    all_evals = get_all_evaluations()
    if not all_evals:
        st.info("No evaluations yet. Run one from the **New Evaluation** tab.")
        st.stop()

    # Evaluation selector
    eval_options = {
        f"#{e['id']} — {e['prompt'][:60]}… ({e['created_at'][:10]})": e["id"]
        for e in all_evals
    }
    selected_label = st.selectbox("Select Evaluation", list(eval_options.keys()))
    eval_id = eval_options[selected_label]

    df = build_scores_dataframe(eval_id)
    if df.empty:
        st.warning("No scores found for this evaluation.")
        st.stop()

    # ── Score gauges ──────────────────────────────────────────────────────
    st.markdown("### 🏆 Weighted Score Rankings")
    auto_df = df[df["Scorer"] == "Automated"]

    if not auto_df.empty:
        ranked = auto_df[["Model", "Weighted ⭐"]].sort_values("Weighted ⭐", ascending=False).reset_index(drop=True)
        gauge_cols = st.columns(len(ranked))
        for i, (col, row) in enumerate(zip(gauge_cols, ranked.itertuples())):
            with col:
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                st.markdown(f"<div style='text-align:center;font-size:1.5rem'>{medal}</div>", unsafe_allow_html=True)
                fig = build_weighted_score_gauge(row.Model, row._3)
                st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")

    # ── Score table ───────────────────────────────────────────────────────
    st.markdown("### 📋 Full Score Table")
    tab1, tab2 = st.tabs(["Automated Scores", "Human Scores"])

    with tab1:
        a_df = df[df["Scorer"] == "Automated"].drop(columns=["Scorer"])
        if a_df.empty:
            st.info("No automated scores found.")
        else:
            st.dataframe(
                a_df.style.background_gradient(
                    cmap="RdYlGn",
                    subset=["Accuracy","Completeness","Clarity","Creativity","Helpfulness","Overall","Weighted ⭐"],
                    vmin=0, vmax=10
                ),
                use_container_width=True,
            )

    with tab2:
        h_df = df[df["Scorer"] == "Human"].drop(columns=["Scorer"])
        if h_df.empty:
            st.info("No human scores yet. Submit them from the New Evaluation tab.")
        else:
            st.dataframe(
                h_df.style.background_gradient(
                    cmap="RdYlGn",
                    subset=["Accuracy","Completeness","Clarity","Creativity","Helpfulness","Overall","Weighted ⭐"],
                    vmin=0, vmax=10
                ),
                use_container_width=True,
            )

    # ── Charts ────────────────────────────────────────────────────────────
    st.markdown("### 📈 Visual Analysis")

    # Build scores_by_model dict from automated scores for radar/bar
    auto_rows = auto_df.to_dict("records") if not auto_df.empty else []
    scores_by_model = {
        r["Model"]: {
            "accuracy": r["Accuracy"], "completeness": r["Completeness"],
            "clarity": r["Clarity"], "creativity": r["Creativity"],
            "helpfulness": r["Helpfulness"], "overall_quality": r["Overall"],
        }
        for r in auto_rows
    }

    if scores_by_model:
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["🕸 Radar", "📊 Bar Chart", "🌡 Heatmap"])

        with chart_tab1:
            st.plotly_chart(build_radar_chart(scores_by_model), use_container_width=True, key="radar")
        with chart_tab2:
            st.plotly_chart(build_bar_chart(scores_by_model), use_container_width=True, key="bar")
        with chart_tab3:
            st.plotly_chart(build_score_heatmap(scores_by_model), use_container_width=True, key="heat")

    # ── Performance metrics ───────────────────────────────────────────────
    st.markdown("### ⚡ Performance Metrics")
    perf_data = []
    for resp in get_responses_for_evaluation(eval_id):
        perf_data.append({
            "Model": resp["model_name"],
            "Latency (ms)": round(resp.get("latency_ms", 0), 1),
            "Tokens Used": resp.get("tokens_used", 0),
        })
    if perf_data:
        perf_df = pd.DataFrame(perf_data).set_index("Model")
        st.dataframe(perf_df, use_container_width=True)

    # ── Historical trend ──────────────────────────────────────────────────
    st.markdown("### 📉 Historical Weighted Scores")
    history_rows = []
    for ev in all_evals:
        for resp in get_responses_for_evaluation(ev["id"]):
            for sc in get_scores_for_response(resp["id"]):
                if sc["scorer_type"] == "automated" and sc.get("weighted_score"):
                    history_rows.append({
                        "created_at":    ev["created_at"][:16],
                        "model_name":    resp["model_name"],
                        "weighted_score":sc["weighted_score"],
                    })

    if len(history_rows) >= 2:
        hist_df = pd.DataFrame(history_rows)
        st.plotly_chart(build_history_line_chart(hist_df), use_container_width=True, key="history")
    else:
        st.caption("Run more evaluations to see trend lines.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📜 History":
    st.markdown('<div class="section-header">📜 Evaluation History</div>', unsafe_allow_html=True)

    search_query = st.text_input("🔍 Search prompts…", placeholder="machine learning, python, …")

    if search_query:
        evals = search_evaluations(search_query)
        st.caption(f"{len(evals)} result(s) for '{search_query}'")
    else:
        evals = get_all_evaluations()
        st.caption(f"{len(evals)} total evaluation(s)")

    if not evals:
        st.info("No evaluations found.")
        st.stop()

    for ev in evals:
        with st.expander(f"#{ev['id']} · {ev['prompt'][:80]} · {ev['created_at'][:16]}"):
            st.markdown(f"**Prompt:** {ev['prompt']}")
            responses = get_responses_for_evaluation(ev["id"])
            if not responses:
                st.caption("No responses stored.")
                continue

            for resp in responses:
                scores = get_scores_for_response(resp["id"])
                auto_score = next((s for s in scores if s["scorer_type"] == "automated"), None)
                human_score = next((s for s in scores if s["scorer_type"] == "human"), None)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{resp['model_name']}**")
                    st.markdown(
                        f'<div class="response-card">{resp["response_text"][:400]}{"…" if len(resp["response_text"])>400 else ""}</div>',
                        unsafe_allow_html=True
                    )
                with col2:
                    if auto_score:
                        ws = auto_score.get("weighted_score", 0) or 0
                        st.markdown(f"🤖 Auto: {score_badge(ws)}", unsafe_allow_html=True)
                    if human_score:
                        ws = human_score.get("weighted_score", 0) or 0
                        st.markdown(f"👤 Human: {score_badge(ws)}", unsafe_allow_html=True)
                    st.caption(f"⏱ {resp.get('latency_ms', 0):.0f} ms")
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📤 Export":
    st.markdown('<div class="section-header">📤 Export Results</div>', unsafe_allow_html=True)

    all_evals = get_all_evaluations()
    if not all_evals:
        st.info("No evaluations to export yet.")
        st.stop()

    export_scope = st.radio("Export Scope", ["Single Evaluation", "All Evaluations"])

    if export_scope == "Single Evaluation":
        eval_options = {
            f"#{e['id']} — {e['prompt'][:60]}… ({e['created_at'][:10]})": e["id"]
            for e in all_evals
        }
        selected_label = st.selectbox("Choose Evaluation", list(eval_options.keys()))
        eval_id = eval_options[selected_label]

        data = get_full_evaluation_data(eval_id)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📄 CSV")
            csv_bytes, csv_name = export_to_csv(data)
            st.download_button(
                label="⬇ Download CSV",
                data=csv_bytes,
                file_name=csv_name,
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            st.markdown("#### 🗂 JSON")
            json_bytes, json_name = export_to_json(data)
            st.download_button(
                label="⬇ Download JSON",
                data=json_bytes,
                file_name=json_name,
                mime="application/json",
                use_container_width=True,
            )

        # Preview
        with st.expander("Preview JSON"):
            st.json(data)

    else:
        st.markdown("Export all evaluations (including responses and scores).")
        all_data = [get_full_evaluation_data(e["id"]) for e in all_evals]

        col1, col2 = st.columns(2)
        with col1:
            csv_bytes, csv_name = export_all_to_csv(all_data)
            st.download_button(
                label="⬇ Download All CSV",
                data=csv_bytes,
                file_name=csv_name,
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            json_bytes, json_name = export_all_to_json(all_data)
            st.download_button(
                label="⬇ Download All JSON",
                data=json_bytes,
                file_name=json_name,
                mime="application/json",
                use_container_width=True,
            )

        st.caption(f"Total evaluations in export: **{len(all_data)}**")
