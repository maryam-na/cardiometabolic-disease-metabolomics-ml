from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization import volcano_plot


DATA = ROOT / "data" / "processed" / "merged_dataset.csv"
RANKED = ROOT / "results" / "ranked_biomarkers.csv"
SELECTED = ROOT / "results" / "selected_biomarkers.csv"

st.title("Biomarker Analysis")
if not DATA.exists():
    st.warning("No processed dataset yet.")
    st.stop()

df = pd.read_csv(DATA)
features = [c for c in df.columns if c != "t2d_status" and pd.api.types.is_numeric_dtype(df[c])]
table, fig = volcano_plot(df, features)
st.plotly_chart(fig, use_container_width=True)

if RANKED.exists():
    table = pd.read_csv(RANKED)
st.dataframe(table.head(100), use_container_width=True)

if SELECTED.exists():
    selected = pd.read_csv(SELECTED)
    st.subheader("Selected Biomarkers")
    st.dataframe(selected, use_container_width=True)

feature = st.selectbox("Inspect biomarker", features)
plot_df = df[[feature, "t2d_status"]].copy()
plot_df["status"] = plot_df["t2d_status"].map({0: "Healthy Control", 1: "T2D"})
st.plotly_chart(px.violin(plot_df, x="status", y=feature, box=True, points="all", template="plotly_white"), use_container_width=True)
