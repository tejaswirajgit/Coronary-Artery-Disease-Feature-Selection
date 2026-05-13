"""Train the deployed Logistic Regression model and persist it for the Streamlit app.

Reproduces the final 8-feature scaled Logistic Regression from the notebook
(Release 2, ``random_state=0``) and writes ``models/model.pkl``,
``models/scaler.pkl``, and ``models/metrics.json``.

Run from the project root:

    python -m src.train

or:

    python src/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import (
    BASELINE_FEATURE_NAMES,
    FEATURE_NAMES,
    get_features_and_target,
    load_and_clean,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed_cleveland.data"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 0


def _score(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main() -> dict:
    df = load_and_clean(DATA_PATH)

    # -------- Baseline: unscaled LogReg on all 13 features --------
    X_base = df.loc[:, BASELINE_FEATURE_NAMES]
    y = df["ca_disease"].to_numpy()
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X_base, y, random_state=RANDOM_STATE
    )
    baseline = LogisticRegression(
        max_iter=2_000_000, fit_intercept=False, random_state=RANDOM_STATE
    )
    baseline.fit(Xb_train, yb_train)
    baseline_metrics = _score(baseline, Xb_test, yb_test)
    baseline_metrics["n_features"] = len(BASELINE_FEATURE_NAMES)

    # -------- Final: scaled LogReg on the 8 selected features --------
    X, _ = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(fit_intercept=False, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    final_metrics = _score(model, X_test_scaled, y_test)
    final_metrics["n_features"] = len(FEATURE_NAMES)

    metrics = {
        **final_metrics,
        "baseline": baseline_metrics,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_names": FEATURE_NAMES,
        "coefficients": dict(zip(FEATURE_NAMES, model.coef_[0].tolist())),
        "random_state": RANDOM_STATE,
    }

    print("=" * 60)
    print("Cleveland heart disease — model comparison")
    print("=" * 60)
    print(f"Train / test sizes : {metrics['n_train']} / {metrics['n_test']}")
    print()
    print(f"{'metric':<12}{'baseline (13 raw)':>22}{'final (8 scaled)':>22}")
    for k in ("accuracy", "precision", "recall", "roc_auc"):
        print(f"{k:<12}{baseline_metrics[k]:>22.4f}{final_metrics[k]:>22.4f}")
    print()
    print(f"Baseline confusion matrix : {baseline_metrics['confusion_matrix']}")
    print(f"Final    confusion matrix : {final_metrics['confusion_matrix']}")
    print()
    print("Final-model feature coefficients (scaled):")
    for name, coef in metrics["coefficients"].items():
        print(f"  {name:<22} {coef:+.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print()
    print(f"Saved {MODELS_DIR / 'model.pkl'}")
    print(f"Saved {MODELS_DIR / 'scaler.pkl'}")
    print(f"Saved {MODELS_DIR / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    main()
