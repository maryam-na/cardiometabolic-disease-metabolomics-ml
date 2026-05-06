from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization import class_distribution, correlation_heatmap, missingness_heatmap, pca_plot


DATA = ROOT / "data" / "processed" / "merged_dataset.csv"

st.title("Data Explorer")
if not DATA.exists():
    st.warning("No processed dataset yet.")
    st.stop()

df = pd.read_csv(DATA)
label = "t2d_status"
features = [c for c in df.columns if c != label and pd.api.types.is_numeric_dtype(df[c])]

st.plotly_chart(class_distribution(df, label), use_container_width=True)
with st.expander("Summary Statistics", expanded=True):
    st.dataframe(df[features].describe().T, use_container_width=True)

tabs = st.tabs(["Missingness", "Correlations", "PCA", "Distributions"])
with tabs[0]:
    st.plotly_chart(missingness_heatmap(df, features[:150]), use_container_width=True)
with tabs[1]:
    st.plotly_chart(correlation_heatmap(df, features), use_container_width=True)
with tabs[2]:
    st.plotly_chart(pca_plot(df[features], df[label]), use_container_width=True)
with tabs[3]:
    feature = st.selectbox("Metabolite", features)
    st.line_chart(df[[feature]])
