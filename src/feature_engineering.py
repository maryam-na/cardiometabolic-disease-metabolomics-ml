from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA


def add_clinical_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    columns = {c.lower(): c for c in out.columns}
    if "triglycerides" in columns and "hdl" in columns:
        out["tg_hdl_ratio"] = out[columns["triglycerides"]] / out[columns["hdl"]].replace(0, pd.NA)
    if "glucose" in columns and "insulin" in columns:
        out["homa_ir_proxy"] = out[columns["glucose"]] * out[columns["insulin"]] / 405
    return out


def pca_embedding(X, n_components: int = 2, random_state: int = 42):
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X), pca
