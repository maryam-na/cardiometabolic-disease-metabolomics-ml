from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:  # Allows raw Workbench parsing before ML dependencies are installed.
    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class TransformerMixin:  # type: ignore[no-redef]
        pass

from src.utils import PROCESSED_DIR, RAW_DIR, detect_label_column, find_candidate_tables, read_table


ID_COLUMNS = ["sample_id", "Sample ID", "SampleID", "mb_sample_id", "local_sample_id", "local_sampleid"]


@dataclass
class LoadedDataset:
    data: pd.DataFrame
    feature_columns: list[str]
    label_column: str
    sample_id_column: str | None


class MissingnessFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.30):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.keep_mask_ = X_df.isna().mean(axis=0) <= self.threshold
        self.feature_names_in_ = list(X_df.columns)
        self.feature_names_out_ = list(X_df.loc[:, self.keep_mask_].columns)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))
        return X_df.loc[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr = X_df.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.drop_columns_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        self.keep_columns_ = [column for column in X_df.columns if column not in self.drop_columns_]
        return self

    def transform(self, X):
        return pd.DataFrame(X).loc[:, self.keep_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_columns_)


def _standardize_label(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.lower()
    return text.str.contains("t2d|diabetes|case|patient|dm", regex=True).astype(int)


def _clean_name(value: object) -> str:
    text = str(value).strip()
    text = text.replace(" ", "_").replace("/", "_")
    return text or "unknown"


def _make_unique(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique = []
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        unique.append(name if counts[name] == 1 else f"{name}__dup{counts[name]}")
    return unique


def parse_workbench_matrix(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    raw.columns = [str(c).strip() for c in raw.columns]
    metabolite_col = next((c for c in raw.columns if c.lower().strip() in {"metabolite_name", "metabolite name"}), raw.columns[0])
    refmet_col = next((c for c in raw.columns if c.lower().strip() == "refmet_name"), None)
    factor_row = raw[raw[metabolite_col].astype(str).str.strip().str.lower() == "factors"]
    if factor_row.empty:
        raise ValueError(f"No Factors row found in {path}")
    sample_cols = [c for c in raw.columns if c not in {metabolite_col, refmet_col}]
    labels = factor_row.iloc[0][sample_cols].astype(str).str.replace("Phenotype:", "", regex=False)

    data = raw[raw[metabolite_col].astype(str).str.strip().str.lower() != "factors"].copy()
    names = data[refmet_col].where(data[refmet_col].notna(), data[metabolite_col]) if refmet_col else data[metabolite_col]
    prefix = path.stem.replace("MSdata_ST003390_", "M")
    data.index = _make_unique([f"{prefix}__{_clean_name(name)}" for name in names])
    numeric = data[sample_cols].apply(pd.to_numeric, errors="coerce")
    transposed = numeric.T
    transposed.index.name = "sample_id"
    transposed = transposed.reset_index()
    transposed["Phenotype"] = labels.values
    return transposed


def load_workbench_matrices(raw_dir: Path = RAW_DIR) -> LoadedDataset | None:
    matrix_paths = sorted(raw_dir.glob("MSdata_ST003390_*.txt"))
    if not matrix_paths:
        return None

    merged: pd.DataFrame | None = None
    phenotype: pd.Series | None = None
    for path in matrix_paths:
        matrix = parse_workbench_matrix(path)
        current_pheno = matrix.set_index("sample_id")["Phenotype"]
        feature_part = matrix.drop(columns=["Phenotype"])
        merged = feature_part if merged is None else merged.merge(feature_part, on="sample_id", how="outer")
        phenotype = current_pheno if phenotype is None else phenotype

    if merged is None or phenotype is None:
        return None
    merged = merged.merge(phenotype.rename("Phenotype"), left_on="sample_id", right_index=True, how="left")
    merged = merged.drop_duplicates(subset=["sample_id"])
    merged["t2d_status"] = _standardize_label(merged["Phenotype"])
    feature_cols = [c for c in merged.columns if c not in {"sample_id", "Phenotype", "t2d_status"}]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PROCESSED_DIR / "merged_dataset.csv", index=False)
    return LoadedDataset(merged, feature_cols, "t2d_status", "sample_id")


def load_and_merge_data(raw_dir: Path = RAW_DIR) -> LoadedDataset:
    workbench = load_workbench_matrices(raw_dir)
    if workbench is not None:
        return workbench

    tables = []
    for path in find_candidate_tables(raw_dir):
        try:
            df = read_table(path)
        except Exception:
            continue
        if df.shape[0] >= 20 and df.shape[1] >= 2:
            tables.append((path, df))

    if not tables:
        raise FileNotFoundError(
            f"No usable raw tables found in {raw_dir}. Run `python -m src.utils download` "
            "or place the ST003390 abundance and metadata files in data/raw."
        )

    labeled = [(p, df, detect_label_column(df.columns)) for p, df in tables]
    labeled = [(p, df, label) for p, df, label in labeled if label is not None]
    if labeled:
        _, merged, label_col = max(labeled, key=lambda item: item[1].shape[1])
    else:
        _, merged = max(tables, key=lambda item: item[1].shape[1])
        label_col = detect_label_column(merged.columns)
        if label_col is None:
            raise ValueError("Could not detect phenotype/label column in raw tables.")

    sample_id_col = next((c for c in ID_COLUMNS if c in merged.columns), None)
    merged = merged.drop_duplicates()
    merged["t2d_status"] = _standardize_label(merged[label_col])

    ignore = set(ID_COLUMNS + [label_col, "t2d_status"])
    feature_cols = []
    for column in merged.columns:
        if column in ignore:
            continue
        numeric = pd.to_numeric(merged[column], errors="coerce")
        if numeric.notna().sum() >= max(10, int(0.5 * len(numeric))):
            merged[column] = numeric
            feature_cols.append(column)

    if not feature_cols:
        raise ValueError("No numeric metabolomics features were detected.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PROCESSED_DIR / "merged_dataset.csv", index=False)
    return LoadedDataset(merged, feature_cols, "t2d_status", sample_id_col)


def remove_outliers_iqr(df: pd.DataFrame, features: list[str], k: float = 3.0) -> pd.DataFrame:
    q1 = df[features].quantile(0.25)
    q3 = df[features].quantile(0.75)
    iqr = q3 - q1
    mask = ~((df[features] < (q1 - k * iqr)) | (df[features] > (q3 + k * iqr))).any(axis=1)
    return df.loc[mask].reset_index(drop=True)


def isolation_forest_outlier_mask(X: pd.DataFrame, contamination: float = 0.03) -> np.ndarray:
    from sklearn.ensemble import IsolationForest

    filled = X.fillna(X.median(numeric_only=True))
    detector = IsolationForest(contamination=contamination, random_state=42)
    return detector.fit_predict(filled) == 1


def clipped_log1p(x):
    return np.log1p(np.clip(x, a_min=0, a_max=None))


def build_preprocessor():
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    return Pipeline(
        steps=[
            ("missingness", MissingnessFilter(threshold=0.30)),
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(clipped_log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
