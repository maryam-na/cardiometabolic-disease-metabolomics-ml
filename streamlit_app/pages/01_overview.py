from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.title("Study Overview")
st.write("Public serum metabolomics cohort for Type 2 Diabetes biomarker discovery and calibrated risk modeling.")

cols = st.columns(4)
cols[0].metric("Study", "ST003390")
cols[1].metric("Subjects", "300")
cols[2].metric("Controls", "200")
cols[3].metric("T2D Cases", "100")

st.subheader("Dataset")
st.markdown(
    """
    - Source: Metabolomics Workbench / NMDR
    - Assay: targeted LC-MS serum metabolomics
    - Cohort: healthy controls and newly diagnosed Type 2 Diabetes patients
    - DOI: `10.21228/M81V7G`
    """
)

merged = ROOT / "data" / "processed" / "merged_dataset.csv"
if merged.exists():
    df = pd.read_csv(merged)
    st.subheader("Merged Matrix")
    st.dataframe(df.head(20), use_container_width=True)
else:
    st.warning("Merged dataset not found. Run `python -m src.modeling` after adding raw files.")
