"""
utils/chart_utils.py
---------------------
Reusable Plotly chart builders for the comparison dashboard.
Keeping chart logic here keeps app.py clean and makes charts testable.
"""

from typing import Any
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

CRITERIA = ["accuracy", "completeness", "clarity", "creativity", "helpfulness", "overall_quality"]
CRITERIA_LABELS = ["Accuracy", "Completeness", "Clarity", "Creativity", "Helpfulness", "Overall Quality"]

# Colour palette – one colour per model (cycles if more than 8 models)
PALETTE = [
    "#4F8EF7", "#F75E4F", "#4FD17A", "#F7C84F",
    "#A04FF7", "#4FF7E8", "#F74FA0", "#C8F74F",
]


def build_radar_chart(scores_by_model: dict[str, dict[str, float]]) -> go.Figure:
    """
    Radar (spider) chart comparing all criteria across models.

    Args:
        scores_by_model: {model_name: {criterion: score}}
    """
    fig = go.Figure()

    for i, (model, scores) in enumerate(scores_by_model.items()):
        values = [scores.get(c, 0) for c in CRITERIA]
        values_closed = values + [values[0]]   # close the polygon
        labels_closed = CRITERIA_LABELS + [CRITERIA_LABELS[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            name=model,
            line_color=PALETTE[i % len(PALETTE)],
            opacity=0.75,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont_size=10)
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(t=40, b=60),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
    )
    return fig


def build_bar_chart(scores_by_model: dict[str, dict[str, float]]) -> go.Figure:
    """
    Grouped bar chart: each criterion is a group, each model is a bar.
    """
    fig = go.Figure()

    for i, (model, scores) in enumerate(scores_by_model.items()):
        fig.add_trace(go.Bar(
            name=model,
            x=CRITERIA_LABELS,
            y=[scores.get(c, 0) for c in CRITERIA],
            marker_color=PALETTE[i % len(PALETTE)],
            text=[f"{scores.get(c, 0):.1f}" for c in CRITERIA],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 11], title="Score (1–10)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(t=20, b=80),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
    )
    return fig


def build_weighted_score_gauge(model_name: str, score: float) -> go.Figure:
    """Gauge chart for a single model's weighted score."""
    colour = (
        "#4FD17A" if score >= 7.5
        else "#F7C84F" if score >= 5.0
        else "#F75E4F"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": model_name, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 10]},
            "bar":  {"color": colour},
            "steps": [
                {"range": [0, 5],   "color": "rgba(247,94,79,0.2)"},
                {"range": [5, 7.5], "color": "rgba(247,200,79,0.2)"},
                {"range": [7.5, 10],"color": "rgba(79,209,122,0.2)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"suffix": "/10", "font": {"size": 28}},
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
    )
    return fig


def build_history_line_chart(history_df: pd.DataFrame) -> go.Figure:
    """
    Line chart showing weighted_score over time for each model.

    Args:
        history_df: columns = [created_at, model_name, weighted_score]
    """
    fig = px.line(
        history_df,
        x="created_at",
        y="weighted_score",
        color="model_name",
        markers=True,
        labels={
            "created_at":    "Date",
            "weighted_score":"Weighted Score",
            "model_name":    "Model",
        },
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        yaxis=dict(range=[0, 10.5]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        margin=dict(t=20, b=80),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
    )
    return fig


def build_score_heatmap(scores_by_model: dict[str, dict[str, float]]) -> go.Figure:
    """
    Heatmap: rows = models, columns = criteria.
    """
    models   = list(scores_by_model.keys())
    z_values = [[scores_by_model[m].get(c, 0) for c in CRITERIA] for m in models]

    fig = go.Figure(go.Heatmap(
        z=z_values,
        x=CRITERIA_LABELS,
        y=models,
        colorscale="RdYlGn",
        zmin=0,
        zmax=10,
        text=[[f"{v:.1f}" for v in row] for row in z_values],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        height=max(250, 80 * len(models)),
        margin=dict(t=20, b=20, l=120, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#E2E8F0",
    )
    return fig
