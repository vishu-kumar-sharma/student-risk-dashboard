"""
Student Analytics & Dropout Prediction Dashboard.

Run after train.py has generated the model artifacts:
    streamlit run app.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
METADATA_PATH = MODEL_DIR / "training_metadata.joblib"
LOCAL_CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned_student_lifestyle_performance_dataset.csv"
FALLBACK_DATA_PATH = Path(
    r"C:\Users\meowk\Downloads\archive\student_lifestyle_performance_dataset.csv"
)
TARGET_COLUMN = "Risk_Level"
RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
RISK_COLORS = {
    "Low Risk": "#16A34A",
    "Medium Risk": "#F59E0B",
    "High Risk": "#DC2626",
}
RISK_CELL_COLORS = {
    "Low Risk": ("#DCFCE7", "#166534"),
    "Medium Risk": ("#FEF3C7", "#92400E"),
    "High Risk": ("#FEE2E2", "#991B1B"),
}
DATE_COLUMN_TOKENS = ("date", "month", "year", "semester", "term", "created", "enrolled")


st.set_page_config(
    page_title="Student Analytics & Dropout Prediction",
    layout="wide",
)


def inject_global_styles() -> None:
    """Apply a modern SaaS-style skin while preserving the Streamlit layout."""
    st.markdown(
        """
        <style>
            :root {
                --surface: #ffffff;
                --surface-soft: #f8fafc;
                --text: #0f172a;
                --muted: #475569;
                --line: rgba(148, 163, 184, 0.24);
                --shadow: 0 18px 45px rgba(15, 23, 42, 0.09);
            }

            .stApp {
                background:
                    radial-gradient(circle at 12% 8%, rgba(20, 184, 166, 0.10), transparent 28%),
                    radial-gradient(circle at 88% 18%, rgba(245, 158, 11, 0.10), transparent 24%),
                    linear-gradient(180deg, #f8fbff 0%, #eef4f8 100%);
                color: var(--text);
                font-family: Inter, "Segoe UI", sans-serif;
            }

            .block-container {
                max-width: 1440px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            [data-testid="stSidebar"] > div:first-child {
                background: linear-gradient(180deg, #0f172a 0%, #164e63 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span {
                color: rgba(255, 255, 255, 0.92) !important;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label {
                border-radius: 14px;
                padding: 0.55rem 0.75rem;
                transition: background 180ms ease, transform 180ms ease;
            }

            [data-testid="stSidebar"] [role="radiogroup"] label:hover {
                background: rgba(255, 255, 255, 0.09);
                transform: translateX(3px);
            }

            .page-header {
                background:
                    linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(240, 253, 250, 0.78)),
                    linear-gradient(135deg, rgba(14, 165, 233, 0.08), rgba(245, 158, 11, 0.07));
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 28px;
                box-shadow: var(--shadow);
                padding: 1.55rem 1.7rem;
                margin-bottom: 1.25rem;
                animation: fadeUp 420ms ease both;
            }

            .eyebrow {
                color: #0f766e;
                font-size: 0.76rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .page-header h1 {
                color: var(--text);
                font-size: clamp(1.7rem, 2.4vw, 2.75rem);
                line-height: 1.08;
                margin: 0;
                letter-spacing: 0;
            }

            .page-header p {
                color: var(--muted);
                font-size: 1rem;
                margin: 0.55rem 0 0;
                max-width: 760px;
            }

            .metric-card {
                --accent: #0ea5e9;
                position: relative;
                min-height: 138px;
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 72%, rgba(20, 184, 166, 0.08) 100%);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 22px;
                box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
                padding: 1.1rem 1.15rem;
                overflow: hidden;
                transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
                animation: fadeUp 480ms ease both;
            }

            .metric-card:hover,
            .stPlotlyChart:hover,
            div[data-testid="stForm"]:hover {
                transform: translateY(-3px);
                box-shadow: 0 22px 55px rgba(15, 23, 42, 0.12);
                border-color: rgba(20, 184, 166, 0.35);
            }

            .metric-card::before {
                content: "";
                position: absolute;
                inset: 0 0 auto 0;
                height: 4px;
                background: var(--accent);
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .metric-value {
                color: var(--text);
                font-size: 2rem;
                font-weight: 850;
                line-height: 1.1;
                margin-top: 0.65rem;
            }

            .metric-helper {
                color: var(--muted);
                font-size: 0.86rem;
                margin-top: 0.45rem;
            }

            .stPlotlyChart,
            div[data-testid="stForm"],
            .list-panel,
            .prediction-card,
            .probability-panel {
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 24px;
                box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
                padding: 0.85rem;
                transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
                animation: fadeUp 500ms ease both;
            }

            div[data-testid="stForm"] {
                padding: 1.35rem;
            }

            h1, h2, h3 {
                letter-spacing: 0;
            }

            h2, h3 {
                color: var(--text);
            }

            .block-container [data-testid="stCaptionContainer"],
            .block-container [data-testid="stCaptionContainer"] *,
            .block-container [data-testid="stMarkdownContainer"] p,
            .block-container div[data-testid="stSelectbox"] label,
            .block-container div[data-testid="stMultiSelect"] label,
            .block-container div[data-testid="stSlider"] label,
            .block-container div[data-testid="stNumberInput"] label {
                color: #334155 !important;
            }

            .list-panel {
                padding: 1.05rem 1.15rem;
            }

            .panel-title {
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                gap: 1rem;
                margin-bottom: 0.85rem;
            }

            .panel-title h3 {
                font-size: 1.05rem;
                margin: 0;
            }

            .panel-title span {
                color: var(--muted);
                font-size: 0.82rem;
                font-weight: 700;
            }

            .top-student-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                border-bottom: 1px solid rgba(148, 163, 184, 0.20);
                padding: 0.78rem 0;
            }

            .top-student-row:last-child {
                border-bottom: 0;
                padding-bottom: 0.15rem;
            }

            .top-student-title {
                color: var(--text);
                font-weight: 800;
                margin-bottom: 0.2rem;
            }

            .branch-tag {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                background: #e0f2fe;
                color: #075985;
                font-size: 0.72rem;
                font-weight: 850;
                margin-left: 0.35rem;
                padding: 0.18rem 0.5rem;
            }

            .top-student-meta {
                color: var(--muted);
                font-size: 0.82rem;
            }

            .top-student-score {
                color: var(--text);
                font-size: 1.2rem;
                font-weight: 850;
                text-align: right;
            }

            .risk-pill {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                white-space: nowrap;
                border-radius: 999px;
                font-size: 0.76rem;
                font-weight: 850;
                padding: 0.25rem 0.62rem;
                border: 1px solid transparent;
            }

            .risk-pill.low-risk {
                color: #166534;
                background: #dcfce7;
                border-color: rgba(22, 163, 74, 0.20);
            }

            .risk-pill.medium-risk {
                color: #92400e;
                background: #fef3c7;
                border-color: rgba(245, 158, 11, 0.24);
            }

            .risk-pill.high-risk {
                color: #991b1b;
                background: #fee2e2;
                border-color: rgba(220, 38, 38, 0.24);
            }

            .prediction-card {
                min-height: 245px;
                display: grid;
                gap: 0.85rem;
                align-content: start;
                padding: 1.2rem;
            }

            .prediction-topline {
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .prediction-risk {
                color: var(--text);
                font-size: 1.75rem;
                font-weight: 900;
                line-height: 1.05;
            }

            .confidence-dial {
                display: grid;
                place-items: center;
                width: 126px;
                height: 126px;
                border-radius: 999px;
                margin: 0.35rem 0 0.1rem;
                position: relative;
                box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
            }

            .confidence-dial::after {
                content: "";
                position: absolute;
                inset: 12px;
                border-radius: inherit;
                background: #ffffff;
                box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.16);
            }

            .confidence-dial span {
                position: relative;
                z-index: 1;
                color: var(--text);
                font-size: 1.7rem;
                font-weight: 900;
            }

            .probability-panel {
                padding: 1.15rem;
            }

            .probability-row {
                margin-top: 0.85rem;
            }

            .probability-label {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 0.35rem;
            }

            .probability-label strong {
                color: var(--text);
                font-size: 0.92rem;
            }

            .progress-track {
                height: 10px;
                overflow: hidden;
                border-radius: 999px;
                background: #e2e8f0;
            }

            .progress-fill {
                height: 100%;
                border-radius: inherit;
                transition: width 550ms ease;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.22);
                box-shadow: 0 14px 35px rgba(15, 23, 42, 0.07);
            }

            .stButton > button,
            div[data-testid="stFormSubmitButton"] button {
                border-radius: 999px;
                border: 0;
                color: white;
                background: linear-gradient(135deg, #0f766e, #2563eb);
                font-weight: 800;
                padding: 0.65rem 1.3rem;
                box-shadow: 0 12px 26px rgba(37, 99, 235, 0.22);
                transition: transform 160ms ease, box-shadow 160ms ease;
            }

            .stButton > button:hover,
            div[data-testid="stFormSubmitButton"] button:hover {
                transform: translateY(-2px);
                box-shadow: 0 16px 34px rgba(37, 99, 235, 0.28);
            }

            @keyframes fadeUp {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @media (max-width: 900px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .metric-card,
                .prediction-card {
                    min-height: auto;
                }

                .top-student-row,
                .panel-title {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .top-student-score {
                    text-align: left;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(eyebrow: str, title: str, description: str) -> None:
    """Render the consistent page heading used by the dashboard pages."""
    st.markdown(
        f"""
        <section class="page-header">
            <div class="eyebrow">{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(description)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, helper: str, accent: str) -> None:
    """Render a high-emphasis metric card."""
    st.markdown(
        f"""
        <div class="metric-card" style="--accent: {accent};">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-helper">{escape(helper)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_style(
    fig: go.Figure,
    *,
    height: int = 360,
    show_legend: bool = True,
) -> go.Figure:
    """Give Plotly charts a shared readable SaaS-dashboard style."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": 'Inter, "Segoe UI", sans-serif', "color": "#1e293b", "size": 13},
        title={"font": {"size": 19, "color": "#0f172a"}, "x": 0.02, "xanchor": "left"},
        margin={"l": 34, "r": 54, "t": 66, "b": 48},
        hoverlabel={"bgcolor": "#0f172a", "font_size": 13, "font_color": "#ffffff"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#334155", "size": 12},
            "title": {"font": {"color": "#334155", "size": 12}},
        },
        showlegend=show_legend,
        uniformtext={"mode": "show", "minsize": 11},
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(100, 116, 139, 0.55)",
        ticks="outside",
        tickcolor="rgba(100, 116, 139, 0.55)",
        tickfont={"color": "#334155", "size": 12},
        title_font={"color": "#334155", "size": 13},
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="rgba(100, 116, 139, 0.22)",
        zeroline=False,
        linecolor="rgba(100, 116, 139, 0.55)",
        ticks="outside",
        tickcolor="rgba(100, 116, 139, 0.55)",
        tickfont={"color": "#334155", "size": 12},
        title_font={"color": "#334155", "size": 13},
        automargin=True,
    )
    return fig


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Load training metadata created by train.py."""
    if not METADATA_PATH.exists():
        st.error("Training metadata was not found. Run `python train.py` first.")
        st.stop()

    return joblib.load(METADATA_PATH)


@st.cache_data(show_spinner=False)
def load_student_data(metadata: dict) -> pd.DataFrame:
    """Load the cleaned directory dataset, falling back to raw CSV if needed."""
    metadata_cleaned_path = Path(metadata.get("cleaned_dataset_path", ""))
    candidate_paths = [
        LOCAL_CLEANED_DATA_PATH,
        metadata_cleaned_path,
        FALLBACK_DATA_PATH,
    ]
    data_path = next((path for path in candidate_paths if path.exists()), None)

    if data_path is None:
        st.error(
            "Student dataset was not found. For deployment, commit "
            "`data/cleaned_student_lifestyle_performance_dataset.csv` with the app."
        )
        st.stop()

    df = pd.read_csv(data_path)
    df.columns = [column.strip() for column in df.columns]

    # If the fallback raw CSV is used, recreate the engineered target for charts.
    if TARGET_COLUMN not in df.columns:
        df = engineer_risk_level(df)

    return df


@st.cache_resource(show_spinner=False)
def load_model(artifact_path: str):
    """Load a persisted sklearn pipeline."""
    metadata_path = Path(artifact_path)
    local_path = MODEL_DIR / metadata_path.name
    path = local_path if local_path.exists() else metadata_path

    if not path.exists():
        st.error(
            f"Model artifact was not found. Expected `{local_path}`. "
            "For deployment, commit the `models/` folder with the app."
        )
        st.stop()

    return joblib.load(path)


def engineer_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the same target logic used during training."""
    output = df.copy()
    cgpa = pd.to_numeric(output["CGPA"], errors="coerce")
    stress = pd.to_numeric(output["Stress_Level_1_to_10"], errors="coerce")

    high_risk = (cgpa < 5.5) | (stress >= 8.0) | ((cgpa < 6.5) & (stress >= 7.0))
    medium_risk = (cgpa < 7.0) | (stress >= 6.0) | ((cgpa < 7.5) & (stress >= 5.5))

    output[TARGET_COLUMN] = np.select(
        [high_risk, medium_risk],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )
    return output


def model_display_name(model_name: str, best_model_name: str) -> str:
    """Add the recommendation label to the best model in dropdowns."""
    if model_name == best_model_name:
        return f"{model_name} (Recommended/Best Model)"
    return model_name


def show_risk_alert(risk_level: str) -> None:
    """Render the prediction with a strong visual color cue."""
    if risk_level == "High Risk":
        st.error(f"Predicted Risk Level: {risk_level}")
    elif risk_level == "Medium Risk":
        st.warning(f"Predicted Risk Level: {risk_level}")
    else:
        st.success(f"Predicted Risk Level: {risk_level}")


def model_metrics_frame(metadata: dict) -> pd.DataFrame:
    """Convert metadata model scores into a chart-ready dataframe."""
    rows = []
    for model_name, model_info in metadata["models"].items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": model_info["accuracy"],
                "Weighted F1": model_info["f1_score_weighted"],
                "Recommended": model_info["is_best_model"],
            }
        )

    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False)


def risk_css_class(risk_level: str) -> str:
    """Return a stable CSS class for a risk label."""
    return risk_level.lower().replace(" ", "-")


def risk_pill_html(risk_level: str) -> str:
    """Render a compact colored risk badge."""
    return (
        f'<span class="risk-pill {risk_css_class(risk_level)}">'
        f"{escape(risk_level)}</span>"
    )


def numeric_series(df: pd.DataFrame, column: str, fallback: float = 0.0) -> pd.Series:
    """Read a numeric column safely for analytics derived from optional fields."""
    if column not in df.columns:
        return pd.Series(fallback, index=df.index, dtype="float64")

    series = pd.to_numeric(df[column], errors="coerce")
    median = series.median()
    if pd.isna(median):
        median = fallback
    return series.fillna(median)


def add_student_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable synthetic student labels when the dataset has no names or IDs."""
    output = df.copy()
    if "Student" in output.columns:
        return output

    labels = []
    for position, index_value in enumerate(output.index):
        try:
            labels.append(f"S-{int(index_value) + 1:04d}")
        except (TypeError, ValueError):
            labels.append(f"S-{position + 1:04d}")
    output.insert(0, "Student", labels)
    return output


def risk_counts_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return ordered risk counts for charts."""
    risk_counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["Risk Level", "Students"]
    return risk_counts


def risk_score_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add a readable composite risk score for ranking student records."""
    output = add_student_labels(df)
    cgpa = numeric_series(output, "CGPA", fallback=7.0).clip(0, 10)
    attendance = numeric_series(output, "Attendance_Percentage", fallback=75.0).clip(0, 100)
    stress = numeric_series(output, "Stress_Level_1_to_10", fallback=5.0).clip(1, 10)
    risk_weight = output[TARGET_COLUMN].map(
        {"Low Risk": 0.18, "Medium Risk": 0.58, "High Risk": 1.0}
    ).fillna(0.35)

    score = (
        (0.44 * risk_weight)
        + (0.23 * (stress / 10))
        + (0.19 * (1 - (cgpa / 10)))
        + (0.14 * (1 - (attendance / 100)))
    ) * 100
    output["Risk Score"] = score.round().astype(int).clip(0, 100)
    return output


def top_at_risk_students(df: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    """Return the highest priority students for the dashboard list."""
    ranked = risk_score_frame(df)
    sort_columns = [TARGET_COLUMN, "Risk Score", "Stress_Level_1_to_10", "CGPA"]
    sort_columns = [column for column in sort_columns if column in ranked.columns]

    ranked["_risk_order"] = ranked[TARGET_COLUMN].map(
        {"High Risk": 3, "Medium Risk": 2, "Low Risk": 1}
    ).fillna(0)
    ranked = ranked.sort_values(
        ["_risk_order", "Risk Score", "Stress_Level_1_to_10", "CGPA"],
        ascending=[False, False, False, True],
    )

    display_columns = [
        "Student",
        "Branch",
        TARGET_COLUMN,
        "CGPA",
        "Attendance_Percentage",
        "Stress_Level_1_to_10",
        "Risk Score",
    ]
    return ranked[[column for column in display_columns if column in ranked.columns]].head(limit)


def build_risk_trend_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Build a risk trend line chart from a date column or ordered cohorts."""
    if df.empty:
        return pd.DataFrame(columns=["Period", "Risk Level", "Student Share"]), "No data"

    work = df.copy()
    period_order = None
    trend_note = "Ordered student cohorts used because the dataset has no date field."

    for column in df.columns:
        if not any(token in column.lower() for token in DATE_COLUMN_TOKENS):
            continue

        parsed_dates = pd.to_datetime(df[column], errors="coerce")
        minimum_valid_dates = max(3, int(len(df) * 0.35))
        if parsed_dates.notna().sum() >= minimum_valid_dates:
            work["_period"] = parsed_dates.dt.to_period("M").dt.to_timestamp()
            trend_note = f"Monthly trend using {column.replace('_', ' ')}."
            break
    else:
        work = work.reset_index(drop=True)
        bucket_count = max(1, min(12, len(work)))
        work["_period_order"] = pd.cut(
            np.arange(len(work)),
            bins=bucket_count,
            labels=False,
            include_lowest=True,
        )
        period_order = [f"Cohort {index + 1}" for index in range(bucket_count)]
        work["_period"] = work["_period_order"].map(lambda value: period_order[int(value)])

    counts = (
        work.groupby(["_period", TARGET_COLUMN], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=RISK_ORDER, fill_value=0)
    )
    if period_order:
        counts = counts.reindex(period_order, fill_value=0)
    else:
        counts = counts.sort_index()

    totals = counts.sum(axis=1).replace(0, np.nan)
    shares = counts.div(totals, axis=0).fillna(0) * 100
    shares.index.name = "Period"
    trend_df = shares.reset_index().melt(
        id_vars="Period",
        var_name="Risk Level",
        value_name="Student Share",
    )
    return trend_df, trend_note


def risk_trend_chart(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Create the risk trend line chart."""
    trend_df, trend_note = build_risk_trend_frame(df)
    fig = px.line(
        trend_df,
        x="Period",
        y="Student Share",
        color="Risk Level",
        markers=True,
        category_orders={"Risk Level": RISK_ORDER},
        color_discrete_map=RISK_COLORS,
        title="Risk Trend Over Time",
    )
    fig.update_traces(line={"width": 3}, marker={"size": 8, "line": {"width": 1, "color": "#ffffff"}})
    fig.update_yaxes(title="Student share", ticksuffix="%", range=[0, 100])
    fig.update_xaxes(title="")
    fig.update_traces(hovertemplate="%{x}<br>%{y:.1f}% %{fullData.name}<extra></extra>")
    return apply_plotly_style(fig, height=380), trend_note


def risk_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create the overall risk distribution donut."""
    risk_counts = risk_counts_frame(df)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=risk_counts["Risk Level"],
                values=risk_counts["Students"],
                hole=0.62,
                sort=False,
                marker={"colors": [RISK_COLORS[risk] for risk in risk_counts["Risk Level"]]},
                textinfo="percent",
                textfont={"size": 14, "color": "#0f172a"},
                hovertemplate="%{label}<br>%{value:,} students<extra></extra>",
            )
        ]
    )
    at_risk_count = int(risk_counts.loc[risk_counts["Risk Level"] != "Low Risk", "Students"].sum())
    total_students = max(int(risk_counts["Students"].sum()), 1)
    at_risk_share = at_risk_count / total_students
    fig.add_annotation(
        text=f"<b>{at_risk_share:.0%}</b><br>At risk",
        showarrow=False,
        font={"size": 18, "color": "#0f172a"},
    )
    fig.update_layout(title="Overall Risk Distribution")
    return apply_plotly_style(fig, height=380, show_legend=True)


def cgpa_attendance_scatter(df: pd.DataFrame) -> go.Figure:
    """Create the CGPA versus attendance scatter plot."""
    scatter_df = add_student_labels(df)
    hover_columns = [
        column
        for column in ["Branch", "Stress_Level_1_to_10", "Internal_Marks"]
        if column in scatter_df.columns
    ]
    fig = px.scatter(
        scatter_df,
        x="CGPA",
        y="Attendance_Percentage",
        color=TARGET_COLUMN,
        size="Stress_Level_1_to_10" if "Stress_Level_1_to_10" in scatter_df.columns else None,
        size_max=18,
        hover_name="Student",
        hover_data=hover_columns,
        category_orders={TARGET_COLUMN: RISK_ORDER},
        color_discrete_map=RISK_COLORS,
        labels={
            "CGPA": "CGPA",
            "Attendance_Percentage": "Attendance",
            TARGET_COLUMN: "Risk Level",
            "Stress_Level_1_to_10": "Stress",
        },
        title="CGPA vs Attendance",
    )
    fig.update_traces(marker={"opacity": 0.78, "line": {"width": 0.8, "color": "#ffffff"}})
    fig.add_vline(
        x=float(numeric_series(scatter_df, "CGPA", fallback=7.0).mean()),
        line_dash="dot",
        line_color="rgba(15, 23, 42, 0.38)",
    )
    fig.add_hline(
        y=float(numeric_series(scatter_df, "Attendance_Percentage", fallback=75.0).mean()),
        line_dash="dot",
        line_color="rgba(15, 23, 42, 0.38)",
    )
    fig.update_xaxes(range=[0, 10])
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return apply_plotly_style(fig, height=440)


def department_risk_chart(df: pd.DataFrame) -> go.Figure:
    """Create a department-wise stacked risk distribution."""
    department_column = "Branch" if "Branch" in df.columns else df.columns[0]
    department_df = (
        df.groupby([department_column, TARGET_COLUMN], observed=False)
        .size()
        .reset_index(name="Students")
        .rename(columns={department_column: "Department", TARGET_COLUMN: "Risk Level"})
    )
    high_risk_order = (
        department_df[department_df["Risk Level"] == "High Risk"]
        .sort_values("Students", ascending=False)["Department"]
        .tolist()
    )
    remaining_departments = [
        department
        for department in sorted(department_df["Department"].unique().tolist())
        if department not in high_risk_order
    ]
    fig = px.bar(
        department_df,
        x="Department",
        y="Students",
        color="Risk Level",
        text="Students",
        barmode="stack",
        category_orders={
            "Department": high_risk_order + remaining_departments,
            "Risk Level": RISK_ORDER,
        },
        color_discrete_map=RISK_COLORS,
        title="Department-Wise Risk Distribution",
    )
    fig.update_traces(textposition="inside", textfont={"color": "#ffffff", "size": 11})
    fig.update_yaxes(title="Students")
    fig.update_xaxes(title="")
    return apply_plotly_style(fig, height=420)


def model_accuracy_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """Create a compact model performance chart."""
    plot_df = metrics_df.copy()
    plot_df["Status"] = np.where(plot_df["Recommended"], "Recommended", "Alternative")
    fig = px.bar(
        plot_df,
        x="Accuracy",
        y="Model",
        orientation="h",
        color="Status",
        text=plot_df["Accuracy"].map(lambda value: f"{value:.1%}"),
        color_discrete_map={"Recommended": "#14B8A6", "Alternative": "#94A3B8"},
        title="Model Accuracy Comparison",
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Accuracy")
    fig.update_yaxes(title="", categoryorder="total ascending")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return apply_plotly_style(fig, height=420, show_legend=False)


def render_top_at_risk_list(df: pd.DataFrame) -> None:
    """Render the top at-risk students as a focused action list."""
    students = top_at_risk_students(df)
    rows = []
    for _, row in students.iterrows():
        student = escape(str(row.get("Student", "Student")))
        branch = escape(str(row.get("Branch", "Unknown")))
        risk_level = str(row.get(TARGET_COLUMN, "Unknown"))
        cgpa = float(row.get("CGPA", 0))
        attendance = float(row.get("Attendance_Percentage", 0))
        stress = float(row.get("Stress_Level_1_to_10", 0))
        score = int(row.get("Risk Score", 0))
        rows.append(
            '<div class="top-student-row">'
            "<div>"
            f'<div class="top-student-title">{student} <span class="branch-tag">{branch}</span></div>'
            f'<div class="top-student-meta">CGPA {cgpa:.2f} | Attendance {attendance:.1f}% | Stress {stress:.1f}/10</div>'
            "</div>"
            "<div>"
            f'<div class="top-student-score">{score}</div>'
            f"{risk_pill_html(risk_level)}"
            "</div>"
            "</div>"
        )

    html = (
        '<div class="list-panel">'
        '<div class="panel-title">'
        "<h3>Top At-Risk Students</h3>"
        "<span>Priority score</span>"
        "</div>"
        f"{''.join(rows)}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_prediction_summary(risk_level: str, confidence: float | None) -> None:
    """Render the model prediction and confidence dial."""
    risk_color = RISK_COLORS.get(risk_level, "#64748B")
    confidence_degrees = 0 if confidence is None else max(0, min(confidence, 1)) * 360
    confidence_label = "N/A" if confidence is None else f"{confidence:.0%}"
    helper_text = (
        "Probability is unavailable for this model."
        if confidence is None
        else "Top class probability for this prediction."
    )
    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-topline">Predicted risk level</div>
            <div>{risk_pill_html(risk_level)}</div>
            <div class="prediction-risk">{escape(risk_level)}</div>
            <div class="confidence-dial" style="background: conic-gradient({risk_color} {confidence_degrees:.1f}deg, #e2e8f0 0deg);">
                <span>{confidence_label}</span>
            </div>
            <div class="metric-helper">{escape(helper_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(probability_df: pd.DataFrame) -> None:
    """Render risk prediction probabilities as colored progress bars."""
    ordered = probability_df.copy()
    ordered["Risk Level"] = pd.Categorical(ordered["Risk Level"], categories=RISK_ORDER, ordered=True)
    ordered = ordered.sort_values("Risk Level")
    rows = []
    for _, row in ordered.iterrows():
        risk_level = str(row["Risk Level"])
        probability = float(row["Probability"])
        color = RISK_COLORS.get(risk_level, "#64748B")
        rows.append(
            '<div class="probability-row">'
            '<div class="probability-label">'
            f"{risk_pill_html(risk_level)}"
            f"<strong>{probability:.1%}</strong>"
            "</div>"
            '<div class="progress-track">'
            f'<div class="progress-fill" style="width: {probability * 100:.1f}%; background: {color};"></div>'
            "</div>"
            "</div>"
        )

    html = (
        '<div class="probability-panel">'
        '<div class="panel-title">'
        "<h3>Prediction Confidence</h3>"
        "<span>Class probability</span>"
        "</div>"
        f"{''.join(rows)}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def probability_chart(probability_df: pd.DataFrame) -> go.Figure:
    """Create a horizontal probability chart for the predictor result."""
    ordered = probability_df.copy()
    ordered["Risk Level"] = pd.Categorical(ordered["Risk Level"], categories=RISK_ORDER, ordered=True)
    ordered = ordered.sort_values("Risk Level")
    fig = px.bar(
        ordered,
        x="Probability",
        y="Risk Level",
        orientation="h",
        color="Risk Level",
        text=ordered["Probability"].map(lambda value: f"{value:.1%}"),
        category_orders={"Risk Level": RISK_ORDER},
        color_discrete_map=RISK_COLORS,
        title="Risk Prediction Probability",
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return apply_plotly_style(fig, height=340, show_legend=False)


def risk_cell_style(value: object) -> str:
    """Color risk labels in Streamlit dataframes."""
    background, color = RISK_CELL_COLORS.get(str(value), ("#E2E8F0", "#334155"))
    return (
        f"background-color: {background}; color: {color}; "
        "font-weight: 800; border-radius: 8px;"
    )


def risk_column_styles(series: pd.Series) -> List[str]:
    """Color an entire risk-label column in a pandas-version-safe way."""
    return [risk_cell_style(value) for value in series]


def styled_student_frame(frame: pd.DataFrame):
    """Return a formatted dataframe styler with risk color coding."""
    formatters = {
        "CGPA": "{:.2f}",
        "Attendance_Percentage": "{:.1f}%",
        "Stress_Level_1_to_10": "{:.1f}",
        "Internal_Marks": "{:.1f}",
        "Risk Score": "{:.0f}",
    }
    formatters = {column: formatter for column, formatter in formatters.items() if column in frame.columns}
    styler = frame.style.format(formatters)
    if TARGET_COLUMN in frame.columns:
        styler = styler.apply(risk_column_styles, subset=[TARGET_COLUMN], axis=0)
    return styler


def dashboard_view(df: pd.DataFrame, metadata: dict) -> None:
    """Dashboard section with KPIs, risk distribution, and model comparison."""
    render_page_header(
        "Student analytics",
        "Student Analytics & Dropout Prediction Dashboard",
        "Monitor academic risk, attendance patterns, model quality, and intervention priorities from one focused dashboard.",
    )

    total_students = len(df)
    average_cgpa = numeric_series(df, "CGPA", fallback=0).mean()
    average_attendance = numeric_series(df, "Attendance_Percentage", fallback=0).mean()
    high_risk_count = int((df[TARGET_COLUMN] == "High Risk").sum())
    medium_risk_count = int((df[TARGET_COLUMN] == "Medium Risk").sum())
    at_risk_count = high_risk_count + medium_risk_count
    at_risk_rate = at_risk_count / total_students if total_students else 0

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Total Students", f"{total_students:,}", "Students in the current analytics dataset.", "#0EA5E9")
    with metric_cols[1]:
        render_metric_card("Average CGPA", f"{average_cgpa:.2f}", "Academic performance baseline.", "#14B8A6")
    with metric_cols[2]:
        render_metric_card("Average Attendance", f"{average_attendance:.1f}%", "Engagement signal across all students.", "#6366F1")
    with metric_cols[3]:
        render_metric_card("At-Risk Students", f"{at_risk_count:,}", f"{at_risk_rate:.1%} medium or high risk.", "#F97316")

    trend_fig, trend_note = risk_trend_chart(df)
    chart_cols = st.columns([1.45, 1])
    with chart_cols[0]:
        st.plotly_chart(trend_fig, use_container_width=True)
        st.caption(trend_note)
    with chart_cols[1]:
        st.plotly_chart(risk_distribution_chart(df), use_container_width=True)

    insight_cols = st.columns([1.42, 0.9])
    with insight_cols[0]:
        st.plotly_chart(cgpa_attendance_scatter(df), use_container_width=True)
    with insight_cols[1]:
        render_top_at_risk_list(df)

    detail_cols = st.columns([1.25, 1])
    metrics_df = model_metrics_frame(metadata)
    with detail_cols[0]:
        st.plotly_chart(department_risk_chart(df), use_container_width=True)
    with detail_cols[1]:
        st.plotly_chart(model_accuracy_chart(metrics_df), use_container_width=True)

    st.subheader("Model Scores")
    model_scores = metrics_df.assign(
        Accuracy=metrics_df["Accuracy"].map(lambda value: f"{value:.2%}"),
        **{"Weighted F1": metrics_df["Weighted F1"].map(lambda value: f"{value:.2%}")},
        Recommended=metrics_df["Recommended"].map({True: "Yes", False: "No"}),
    )
    st.dataframe(
        model_scores,
        use_container_width=True,
        hide_index=True,
    )


def numeric_default(df: pd.DataFrame, column: str) -> float:
    """Use the dataset median as a practical default for form inputs."""
    return float(pd.to_numeric(df[column], errors="coerce").median())


def build_predictor_inputs(df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, object]:
    """Create Streamlit widgets for the model feature set."""
    inputs: Dict[str, object] = {}

    left_col, right_col = st.columns(2)

    with left_col:
        inputs["Age"] = st.number_input("Age", min_value=15, max_value=60, value=22, step=1)
        inputs["Study_Hours_per_Day"] = st.slider(
            "Study Hours per Day", 0.0, 12.0, numeric_default(df, "Study_Hours_per_Day"), 0.1
        )
        inputs["Sleep_Hours"] = st.slider(
            "Sleep Hours", 0.0, 12.0, numeric_default(df, "Sleep_Hours"), 0.1
        )
        inputs["Screen_Time_Hours"] = st.slider(
            "Screen Time Hours", 0.0, 16.0, numeric_default(df, "Screen_Time_Hours"), 0.1
        )
        inputs["Gym_Hours_per_Week"] = st.slider(
            "Gym Hours per Week", 0.0, 25.0, numeric_default(df, "Gym_Hours_per_Week"), 0.1
        )

    with right_col:
        inputs["Attendance_Percentage"] = st.slider(
            "Attendance Percentage", 0.0, 100.0, numeric_default(df, "Attendance_Percentage"), 0.1
        )
        inputs["Stress_Level_1_to_10"] = st.slider(
            "Stress Level", 1.0, 10.0, numeric_default(df, "Stress_Level_1_to_10"), 0.1
        )
        inputs["Internal_Marks"] = st.slider(
            "Internal Marks", 0.0, 100.0, numeric_default(df, "Internal_Marks"), 0.1
        )
        inputs["CGPA"] = st.slider("CGPA", 0.0, 10.0, numeric_default(df, "CGPA"), 0.01)

    categorical_columns = [column for column in feature_columns if column not in inputs]
    if categorical_columns:
        st.subheader("Student Profile")
        category_cols = st.columns(len(categorical_columns))
        for index, column in enumerate(categorical_columns):
            values = sorted(df[column].dropna().astype(str).unique().tolist())
            inputs[column] = category_cols[index].selectbox(column.replace("_", " "), values)

    return inputs


def predictor_view(df: pd.DataFrame, metadata: dict) -> None:
    """Predictor section with model selection and a student input form."""
    render_page_header(
        "Risk predictor",
        "Predict Student Risk",
        "Estimate a student's risk level and inspect the model confidence behind the prediction.",
    )

    model_names = list(metadata["models"].keys())
    best_model_name = metadata["best_model_name"]
    best_index = model_names.index(best_model_name)

    model_cols = st.columns([1.45, 0.9, 0.9])
    with model_cols[0]:
        selected_model_name = st.selectbox(
            "Choose prediction model",
            model_names,
            index=best_index,
            format_func=lambda name: model_display_name(name, best_model_name),
        )

    selected_info = metadata["models"][selected_model_name]
    with model_cols[1]:
        render_metric_card(
            "Model Accuracy",
            f"{selected_info['accuracy']:.1%}",
            "Validation accuracy for selected model.",
            "#14B8A6",
        )
    with model_cols[2]:
        render_metric_card(
            "Weighted F1",
            f"{selected_info['f1_score_weighted']:.1%}",
            "Class-balanced performance quality.",
            "#6366F1",
        )

    with st.form("student_prediction_form"):
        input_values = build_predictor_inputs(df, metadata["feature_columns"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        model = load_model(selected_info["artifact_path"])
        input_df = pd.DataFrame([input_values], columns=metadata["feature_columns"])
        prediction = model.predict(input_df)[0]

        confidence = None
        probability_df = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            probability_df = pd.DataFrame(
                {
                    "Risk Level": model.classes_,
                    "Probability": probabilities,
                }
            ).sort_values("Probability", ascending=False)
            matching_prediction = probability_df[probability_df["Risk Level"] == prediction]
            confidence = (
                float(matching_prediction["Probability"].iloc[0])
                if not matching_prediction.empty
                else float(probability_df["Probability"].max())
            )

        result_cols = st.columns([0.85, 1.15])
        with result_cols[0]:
            render_prediction_summary(prediction, confidence)
            if probability_df is not None:
                render_probability_bars(probability_df)
        with result_cols[1]:
            if probability_df is not None:
                st.plotly_chart(probability_chart(probability_df), use_container_width=True)
            else:
                show_risk_alert(prediction)


def directory_view(df: pd.DataFrame) -> None:
    """Interactive student directory with simple filters."""
    render_page_header(
        "Student table",
        "Student Directory",
        "Filter, scan, and compare individual students with color-coded risk labels.",
    )

    filter_cols = st.columns(3)
    risk_filter = filter_cols[0].multiselect("Risk Level", RISK_ORDER, default=RISK_ORDER)
    branch_filter = filter_cols[1].multiselect(
        "Branch", sorted(df["Branch"].dropna().unique().tolist())
    )
    residence_filter = filter_cols[2].multiselect(
        "Residence", sorted(df["Residence"].dropna().unique().tolist())
    )

    filtered_df = df[df[TARGET_COLUMN].isin(risk_filter)].copy()

    if branch_filter:
        filtered_df = filtered_df[filtered_df["Branch"].isin(branch_filter)]
    if residence_filter:
        filtered_df = filtered_df[filtered_df["Residence"].isin(residence_filter)]

    filtered_df = risk_score_frame(filtered_df)
    display_columns = [
        "Student",
        "Branch",
        TARGET_COLUMN,
        "Risk Score",
        "CGPA",
        "Attendance_Percentage",
        "Stress_Level_1_to_10",
        "Internal_Marks",
        "Study_Hours_per_Day",
        "Sleep_Hours",
        "Screen_Time_Hours",
        "Gym_Hours_per_Week",
        "Diet_Type",
        "Residence",
        "Age",
    ]
    available_columns = [column for column in display_columns if column in filtered_df.columns]

    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} students")
    st.dataframe(
        styled_student_frame(filtered_df[available_columns]),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """Streamlit app entry point."""
    inject_global_styles()
    metadata = load_metadata()
    df = load_student_data(metadata)

    st.sidebar.title("Student Risk Suite")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Predictor", "Student Directory"],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Recommended model: {metadata['best_model_name']}")
    st.sidebar.caption(f"Last trained: {metadata['trained_at_utc']}")

    if page == "Dashboard":
        dashboard_view(df, metadata)
    elif page == "Predictor":
        predictor_view(df, metadata)
    else:
        directory_view(df)


if __name__ == "__main__":
    main()
