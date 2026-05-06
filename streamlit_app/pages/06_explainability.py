from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SHAP = ROOT / "results" / "shap_feature_ranking.csv"
IMPORTANCE = ROOT / "results" / "feature_importance.csv"

st.title("Explainability")
if SHAP.exists():
    shap_df = pd.read_csv(SHAP)
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    top_n = st.slider("Top features", min_value=10, max_value=50, value=30, step=5)
    nonzero = st.toggle("Show non-zero features only", value=True)
    chart_df = shap_df[shap_df["mean_abs_shap"] > 0] if nonzero else shap_df
    st.plotly_chart(
        px.bar(
            chart_df.head(top_n).sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            template="plotly_white",
            labels={"mean_abs_shap": "Mean absolute SHAP", "feature": "Metabolite"},
        ),
        use_container_width=True,
    )
    st.dataframe(chart_df, use_container_width=True)
elif IMPORTANCE.exists():
    imp = pd.read_csv(IMPORTANCE)
    imp = imp.sort_values("importance", ascending=False)
    top_n = st.slider("Top features", min_value=10, max_value=50, value=30, step=5)
    nonzero = st.toggle("Show non-zero features only", value=True)
    chart_df = imp[imp["importance"] > 0] if nonzero else imp
    if chart_df.empty:
        st.warning("All saved model importance values are zero. Retrain with a larger feature set or generate SHAP artifacts.")
    else:
        cols = st.columns(3)
        cols[0].metric("Features Displayed", len(chart_df))
        cols[1].metric("Max Importance", f"{chart_df['importance'].max():.4f}")
        cols[2].metric("Zero Importance Hidden", int((imp["importance"] <= 0).sum()))
        st.info("This chart uses model feature importance. SHAP values were not generated in the latest training run.")
        st.plotly_chart(
            px.bar(
                chart_df.head(top_n).sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                template="plotly_white",
                labels={"importance": "XGBoost feature importance", "feature": "Metabolite"},
            ),
            use_container_width=True,
        )
    st.dataframe(chart_df, use_container_width=True)
else:
    st.warning("No explainability artifacts found. Train the model first.")
