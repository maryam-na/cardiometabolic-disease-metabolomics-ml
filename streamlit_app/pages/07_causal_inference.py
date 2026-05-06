from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causal_inference import inverse_probability_weighting


DATA = ROOT / "data" / "processed" / "merged_dataset.csv"

st.title("Causal Inference")
if not DATA.exists():
    st.warning("No processed dataset yet.")
    st.stop()

df = pd.read_csv(DATA)
features = [c for c in df.columns if c != "t2d_status" and pd.api.types.is_numeric_dtype(df[c])]
treatment = st.selectbox("Biomarker exposure", features)
covariates = st.multiselect("Adjustment covariates", [c for c in features if c != treatment], default=[])

if st.button("Estimate IPW Effect", type="primary"):
    estimate = inverse_probability_weighting(df, treatment=treatment, covariates=covariates)
    cols = st.columns(3)
    cols[0].metric("ATE", f"{estimate['ate']:.3f}")
    cols[1].metric("CI Low", f"{estimate['ci_low']:.3f}")
    cols[2].metric("CI High", f"{estimate['ci_high']:.3f}")
    st.json(estimate)
