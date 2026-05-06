from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = ROOT / "models" / "xgboost_calibrated.joblib"
DATA = ROOT / "data" / "processed" / "merged_dataset.csv"

st.title("Risk Prediction")
if not MODEL.exists() or not DATA.exists():
    st.warning("Train the calibrated model before using prediction.")
    st.stop()

model = joblib.load(MODEL)
df = pd.read_csv(DATA)
features = [c for c in df.columns if c != "t2d_status" and pd.api.types.is_numeric_dtype(df[c])]

mode = st.segmented_control("Input Mode", ["Existing subject", "Manual values"], default="Existing subject")
if mode == "Existing subject":
    idx = st.number_input("Subject row", min_value=0, max_value=len(df) - 1, value=0, step=1)
    X = df.loc[[idx], features]
else:
    defaults = df[features].median(numeric_only=True)
    values = {}
    selected = st.multiselect("Edit metabolites", features, default=features[: min(8, len(features))])
    for feature in selected:
        values[feature] = st.number_input(feature, value=float(defaults[feature]))
    X = pd.DataFrame([defaults.to_dict()])
    for feature, value in values.items():
        X.loc[0, feature] = value

prob = float(model.predict_proba(X[features])[:, 1][0])
st.metric("Calibrated T2D Probability", f"{prob:.1%}")
st.progress(min(max(prob, 0.0), 1.0))
st.dataframe(X[features].T.rename(columns={X.index[0]: "value"}), use_container_width=True)
