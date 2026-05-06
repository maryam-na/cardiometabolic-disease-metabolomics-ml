from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.decomposition import PCA


def class_distribution(df: pd.DataFrame, label: str = "t2d_status"):
    counts = df[label].map({0: "Healthy Control", 1: "T2D"}).value_counts().reset_index()
    counts.columns = ["class", "count"]
    return px.bar(counts, x="class", y="count", color="class", template="plotly_white")


def missingness_heatmap(df: pd.DataFrame, features: list[str]):
    miss = df[features].isna().astype(int)
    return px.imshow(miss.T, aspect="auto", color_continuous_scale="Greys", template="plotly_white")


def correlation_heatmap(df: pd.DataFrame, features: list[str], max_features: int = 60):
    subset = features[:max_features]
    corr = df[subset].corr(numeric_only=True)
    return px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="plotly_white")


def pca_plot(X: pd.DataFrame, y: pd.Series):
    filled = X.fillna(X.median(numeric_only=True)).fillna(0)
    coords = PCA(n_components=2, random_state=42).fit_transform(filled)
    plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "status": y.map({0: "Healthy Control", 1: "T2D"})})
    return px.scatter(plot_df, x="PC1", y="PC2", color="status", template="plotly_white")


def volcano_plot(df: pd.DataFrame, features: list[str], label: str = "t2d_status"):
    rows = []
    case = df[label] == 1
    for feature in features:
        a = pd.to_numeric(df.loc[case, feature], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[~case, feature], errors="coerce").dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        stat = stats.ttest_ind(np.log1p(a.clip(lower=0)), np.log1p(b.clip(lower=0)), equal_var=False)
        log2fc = np.log2((a.mean() + 1e-9) / (b.mean() + 1e-9))
        rows.append({"feature": feature, "log2fc": log2fc, "p_value": stat.pvalue})
    table = pd.DataFrame(rows)
    if table.empty:
        return table, go.Figure()
    table["neg_log10_p"] = -np.log10(table["p_value"].clip(lower=1e-300))
    fig = px.scatter(table, x="log2fc", y="neg_log10_p", hover_name="feature", template="plotly_white")
    return table.sort_values("p_value"), fig
