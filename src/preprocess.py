"""Data loading, cleaning, and feature selection for the Cleveland heart disease dataset.

Mirrors the cleaning pipeline established in
``notebooks/coronary-artery-disease-feature-selection.ipynb``:

1. Read the 14 raw UCI columns.
2. Coerce ``?`` sentinels to NaN via ``pd.to_numeric(errors='coerce')``.
3. Median-impute ``num_major_vessels`` and ``thallium_scint`` (the only columns
   with missing values in this dataset).
4. Remap ``thallium_scint`` so the codes are ordered by clinical severity
   (normal < reversible defect < fixed defect).
5. Binarise the target ``ca_disease`` (>0 -> 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS: list[str] = [
    "age",
    "sex",
    "chest_pain_type",
    "rest_blood_press",
    "cholesterol",
    "fasting_blood_sugar",
    "rest_ecg",
    "max_heart_rate",
    "exer_ind_angina",
    "st_depression",
    "st_slope",
    "num_major_vessels",
    "thallium_scint",
    "ca_disease",
]

FEATURE_NAMES: list[str] = [
    "sex",
    "chest_pain_type",
    "max_heart_rate",
    "exer_ind_angina",
    "st_depression",
    "st_slope",
    "num_major_vessels",
    "thallium_scint",
]

# All 13 predictors used by the unscaled baseline model.
BASELINE_FEATURE_NAMES: list[str] = [c for c in RAW_COLUMNS if c != "ca_disease"]

TARGET_NAME: str = "ca_disease"

# UCI raw thal codes -> ordered severity codes used by the model.
# 3 = normal, 6 = fixed defect, 7 = reversible defect.
THAL_MAPPING: dict[int, int] = {3: 0, 7: 1, 6: 2}


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Load the raw Cleveland CSV and return a fully cleaned DataFrame.

    The returned frame has 303 rows, all numeric dtypes, no missing values,
    a remapped ``thallium_scint`` column, and a binary ``ca_disease`` target.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        index_col=None,
        usecols=list(range(14)),
        sep=",",
        skipinitialspace=True,
    )
    df = df.apply(pd.to_numeric, errors="coerce")

    for col in ("num_major_vessels", "thallium_scint"):
        df[col] = df[col].fillna(df[col].median())

    df["thallium_scint"] = df["thallium_scint"].replace(THAL_MAPPING)
    df["ca_disease"] = (df["ca_disease"] > 0).astype(int)
    return df


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the 8 selected features and the binary target."""
    X = df.loc[:, FEATURE_NAMES].copy()
    y = df[TARGET_NAME].to_numpy()
    return X, y
