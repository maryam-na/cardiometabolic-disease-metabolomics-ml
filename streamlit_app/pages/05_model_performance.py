from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"

st.title("Model Performance")
metrics_path = RESULTS / "test_metrics.json"
if not metrics_path.exists():
    st.warning("No model metrics found.")
    st.stop()

metrics = json.loads(metrics_path.read_text())
cols = st.columns(4)
for col, key in zip(cols, ["roc_auc", "pr_auc", "f1", "balanced_accuracy"]):
    col.metric(key.replace("_", " ").title(), f"{metrics[key]:.3f}")
cols = st.columns(3)
for col, key in zip(cols, ["sensitivity", "specificity", "brier_score"]):
    col.metric(key.replace("_", " ").title(), f"{metrics[key]:.3f}")

for name, x, y in [
    ("roc_curve.csv", "fpr", "tpr"),
    ("pr_curve.csv", "recall", "precision"),
    ("calibration_curve.csv", "predicted", "observed"),
]:
    path = RESULTS / name
    if path.exists():
        curve = pd.read_csv(path)
        st.plotly_chart(px.line(curve, x=x, y=y, template="plotly_white"), use_container_width=True)
