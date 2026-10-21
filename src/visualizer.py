"""Interactive Plotly visualizations for repository health and model metrics.

Provides clean, publication-ready charts for data exploration, model evaluation,
and real-time repository health diagnosis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Color scheme representing Open Source Health
CLASS_COLORS = {
    "Healthy": "#22c55e",         # Green
    "At Risk": "#eab308",         # Amber/Yellow
    "Likely Abandoned": "#ef4444", # Red
}


def plot_target_distribution(y: pd.Series) -> go.Figure:
    """Create an interactive donut chart showing health status class distribution."""
    counts = y.value_counts().reset_index()
    counts.columns = ["Health Status", "Count"]

    colors = [CLASS_COLORS.get(status, "#64748b") for status in counts["Health Status"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["Health Status"],
                values=counts["Count"],
                hole=0.45,
                marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                textinfo="label+percent+value",
                hoverinfo="label+value+percent",
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Distribution of Repository Health Classes", font=dict(size=18)),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        height=380,
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str]) -> go.Figure:
    """Create an interactive correlation matrix heatmap for numeric features."""
    corr = df[feature_cols].corr().round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Corr"),
        )
    )
    fig.update_layout(
        title=dict(text="Feature Correlation Heatmap", font=dict(size=18)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=520,
        xaxis=dict(tickangle=-45),
    )
    return fig


def plot_feature_boxplots(df: pd.DataFrame, feature: str, target_col: str = "health_status") -> go.Figure:
    """Create a box plot of a numeric feature grouped by health status."""
    color_map = CLASS_COLORS

    fig = px.box(
        df,
        x=target_col,
        y=feature,
        color=target_col,
        color_discrete_map=color_map,
        points="outliers",
        title=f"Distribution of '{feature}' across Health Classes",
        labels={target_col: "Health Status", feature: feature.replace("_", " ").title()},
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
        showlegend=False,
    )
    return fig


def plot_scatter_overview(
    df: pd.DataFrame,
    x_col: str = "days_since_last_commit",
    y_col: str = "stars_count",
    target_col: str = "health_status",
) -> go.Figure:
    """Create a scatter plot of two features colored by health status."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=target_col,
        color_discrete_map=CLASS_COLORS,
        hover_data=[c for c in ["repo_name", "language", "forks_count"] if c in df.columns],
        title=f"Repository Landscape: {y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}",
        labels={
            x_col: x_col.replace("_", " ").title(),
            y_col: y_col.replace("_", " ").title(),
            target_col: "Health Status",
        },
        opacity=0.75,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=430,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Create a grouped bar chart comparing performance across models."""
    metrics = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]

    fig = go.Figure()
    for metric in metrics:
        if metric in comparison_df.columns:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=comparison_df["Model"],
                    y=comparison_df[metric],
                    text=comparison_df[metric].round(3),
                    textposition="auto",
                )
            )

    fig.update_layout(
        barmode="group",
        title=dict(text="Model Performance Comparison (Held-Out Test Set)", font=dict(size=18)),
        yaxis=dict(range=[0, 1.05], title="Score"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], model_name: str) -> go.Figure:
    """Create an annotated confusion matrix heatmap."""
    z = cm
    x = classes
    y = classes

    z_text = [[str(val) for val in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Blues",
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        title=dict(text=f"Confusion Matrix: {model_name}", font=dict(size=18)),
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label", autorange="reversed"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
    )
    return fig


def plot_feature_importance(feature_importance_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Create a horizontal bar chart showing Random Forest feature importances."""
    top_df = feature_importance_df.head(top_n).iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=top_df["Importance"],
            y=[f.replace("_", " ").title() for f in top_df["Feature"]],
            orientation="h",
            marker=dict(
                color=top_df["Importance"],
                colorscale="Viridis",
            ),
            text=top_df["Importance"].round(3),
            textposition="auto",
        )
    )
    fig.update_layout(
        title=dict(text=f"Top {top_n} Feature Importances (Random Forest)", font=dict(size=18)),
        xaxis=dict(title="Importance Weight"),
        yaxis=dict(title="Feature"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=420,
    )
    return fig


def plot_health_score_gauge(score: int, status: str) -> go.Figure:
    """Create an interactive gauge indicator for repository health score."""
    color = CLASS_COLORS.get(status, "#3b82f6")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>Health Score</b><br><span style='font-size:16px;color:{color}'>{status}</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 39], "color": "#fee2e2"},
                    {"range": [40, 69], "color": "#fef9c3"},
                    {"range": [70, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=25, r=25, t=40, b=20),
    )
    return fig


def plot_pillar_breakdown(pillar_scores: Dict[str, Dict[str, int]]) -> go.Figure:
    """Create a horizontal breakdown bar chart showing scores per health pillar."""
    pillars = list(pillar_scores.keys())
    scores = [pillar_scores[p]["score"] for p in pillars]
    maxes = [pillar_scores[p]["max"] for p in pillars]
    percentages = [round((s / m) * 100, 1) for s, m in zip(scores, maxes)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=pillars,
            x=percentages,
            orientation="h",
            marker=dict(
                color=["#3b82f6", "#8b5cf6", "#ec4899", "#10b981"],
            ),
            text=[f"{s}/{m} pts ({pct}%)" for s, m, pct in zip(scores, maxes, percentages)],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=dict(text="Health Score Breakdown by Pillar", font=dict(size=16)),
        xaxis=dict(title="Achievement (% of Max)", range=[0, 105]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=260,
    )
    return fig
