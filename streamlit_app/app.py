from __future__ import annotations

from pathlib import Path

import streamlit as st

from bootstrap import ROOT

st.set_page_config(
    page_title="T2D Metabolomics Biomarker Discovery",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7f8fa; }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e4e7ec; }
    h1, h2, h3 { color: #111827; letter-spacing: 0; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("T2D Serum Metabolomics Risk Modeling")
st.caption("ST003390 targeted LC-MS cohort: 200 healthy controls and 100 newly diagnosed T2D patients")

left, right = st.columns([1.1, 0.9], gap="large")
with left:
    st.subheader("Project Workflow")
    st.write(
        "This dashboard sits on top of a leakage-aware machine learning pipeline for preprocessing, "
        "feature selection, calibrated prediction, explainability, biomarker ranking, and causal sensitivity analysis."
    )
    st.page_link("pages/01_overview.py", label="Overview")
    st.page_link("pages/02_data_explorer.py", label="Data Explorer")
    st.page_link("pages/03_risk_prediction.py", label="Risk Prediction")
    st.page_link("pages/04_biomarker_analysis.py", label="Biomarker Analysis")
    st.page_link("pages/05_model_performance.py", label="Model Performance")
    st.page_link("pages/06_explainability.py", label="Explainability")
    st.page_link("pages/07_causal_inference.py", label="Causal Inference")

with right:
    st.subheader("Run Status")
    required = [
        ROOT / "data" / "processed" / "merged_dataset.csv",
        ROOT / "results" / "test_metrics.json",
        ROOT / "models" / "xgboost_calibrated.joblib",
    ]
    for path in required:
        st.checkbox(path.name, value=path.exists(), disabled=True)
    st.info("Train the model with `python -m src.modeling` after downloading or placing ST003390 files in `data/raw/`.")
