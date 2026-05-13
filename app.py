"""Streamlit interface for the Cleveland heart disease classifier.

Entry point for Hugging Face Spaces. Loads the trained Logistic Regression
model and scaler from ``models/`` and exposes the 8 selected features via
human-readable widgets.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocess import FEATURE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="Coronary Artery Disease Risk Estimator",
    page_icon="❤️",
    layout="wide",
)

# Human-readable label -> raw numeric code used during training.
SEX_OPTIONS = {"Female": 0, "Male": 1}
CHEST_PAIN_OPTIONS = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4,
}
ANGINA_OPTIONS = {"No": 0, "Yes": 1}
ST_SLOPE_OPTIONS = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
# After THAL_MAPPING the training codes are 0=normal, 1=reversible, 2=fixed.
THAL_OPTIONS = {"Normal": 0, "Fixed defect": 2, "Reversible defect": 1}


@st.cache_resource
def load_artifacts() -> tuple[object, object, dict]:
    model = joblib.load(MODELS_DIR / "model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    return model, scaler, metrics


def _risk_color(prob: float) -> str:
    if prob < 0.30:
        return "#16a34a"  # green
    if prob < 0.60:
        return "#d97706"  # amber
    return "#dc2626"  # red


def _interpret(prob: float) -> str:
    pct = prob * 100
    if prob < 0.30:
        return (
            f"The model estimates a **low** probability ({pct:.1f}%) of coronary "
            "artery disease for this feature combination."
        )
    if prob < 0.60:
        return (
            f"The model estimates a **moderate** probability ({pct:.1f}%) of "
            "coronary artery disease. Clinical follow-up may be warranted."
        )
    return (
        f"The model estimates a **high** probability ({pct:.1f}%) of coronary "
        "artery disease. This is an educational estimate only — not a diagnosis."
    )


def _disclaimer() -> None:
    st.error(
        "**Educational project — not medical advice.** "
        "This tool uses a model trained on 303 records from the 1988 UCI "
        "Cleveland dataset. It is not clinically validated and must not be "
        "used for diagnosis, treatment, or any medical decision."
    )


def _sidebar(metrics: dict, model) -> None:
    st.sidebar.header("Model")
    st.sidebar.markdown(
        f"""
- **Algorithm**: Logistic Regression (scaled)
- **Train / test split**: {metrics['n_train']} / {metrics['n_test']}
- **Accuracy**: {metrics['accuracy']:.3f}
- **ROC-AUC**: {metrics['roc_auc']:.3f}
- **Precision**: {metrics['precision']:.3f}
- **Recall**: {metrics['recall']:.3f}
"""
    )

    st.sidebar.header("Feature selection")
    st.sidebar.markdown(
        "Four methods were compared in the notebook to pick the final 8 features:\n"
        "1. statsmodels Logit coefficient significance\n"
        "2. mlxtend exhaustive univariate search\n"
        "3. SelectKBest (`f_classif` + `chi2`)\n"
        "4. Decision Tree / Random Forest importances\n\n"
        "Features that scored highly across at least two methods were kept."
    )

    st.sidebar.header("Feature importance")
    coef = pd.Series(metrics["coefficients"]).sort_values()
    fig, ax = plt.subplots(figsize=(4, 3.2))
    colors = ["#dc2626" if c > 0 else "#2563eb" for c in coef.values]
    ax.barh(coef.index, coef.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Scaled coefficient")
    ax.tick_params(labelsize=8)
    ax.set_title("Logistic regression weights", fontsize=10)
    fig.tight_layout()
    st.sidebar.pyplot(fig)
    st.sidebar.caption(
        "Red = increases predicted risk, blue = decreases. Coefficients act "
        "on standardised inputs."
    )


def _inputs() -> dict[str, float]:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient profile")
        sex = st.radio("Sex", list(SEX_OPTIONS), horizontal=True)
        chest_pain = st.selectbox("Chest pain type", list(CHEST_PAIN_OPTIONS))
        exer_ind_angina = st.radio(
            "Exercise-induced angina", list(ANGINA_OPTIONS), horizontal=True
        )
        max_heart_rate = st.slider(
            "Maximum heart rate achieved (bpm)",
            min_value=70,
            max_value=210,
            value=150,
            step=1,
        )

    with col2:
        st.subheader("Stress-test findings")
        st_slope = st.selectbox(
            "Slope of peak exercise ST segment", list(ST_SLOPE_OPTIONS)
        )
        st_depression = st.slider(
            "ST depression induced by exercise (mm)",
            min_value=0.0,
            max_value=6.5,
            value=1.0,
            step=0.1,
        )
        num_major_vessels = st.slider(
            "Number of major vessels coloured by fluoroscopy",
            min_value=0,
            max_value=3,
            value=0,
            step=1,
        )
        thal = st.selectbox("Thallium scintigraphy result", list(THAL_OPTIONS))

    return {
        "sex": SEX_OPTIONS[sex],
        "chest_pain_type": CHEST_PAIN_OPTIONS[chest_pain],
        "max_heart_rate": max_heart_rate,
        "exer_ind_angina": ANGINA_OPTIONS[exer_ind_angina],
        "st_depression": st_depression,
        "st_slope": ST_SLOPE_OPTIONS[st_slope],
        "num_major_vessels": num_major_vessels,
        "thallium_scint": THAL_OPTIONS[thal],
    }


def main() -> None:
    model, scaler, metrics = load_artifacts()

    st.title("❤️ Coronary Artery Disease Risk Estimator")
    st.caption(
        "Logistic regression trained on the UCI Cleveland heart disease "
        "dataset (303 patients, 14 features). See the sidebar for model "
        "metrics and feature selection methodology."
    )
    _disclaimer()
    _sidebar(metrics, model)

    feature_values = _inputs()
    st.divider()

    if st.button("Predict risk", type="primary", use_container_width=True):
        row = pd.DataFrame([[feature_values[name] for name in FEATURE_NAMES]],
                           columns=FEATURE_NAMES)
        row_scaled = scaler.transform(row)
        prob = float(model.predict_proba(row_scaled)[0, 1])

        st.subheader("Estimated probability")
        color = _risk_color(prob)
        st.markdown(
            f"""
<div style="background:#1f2937;border-radius:8px;padding:14px;">
  <div style="font-size:34px;color:{color};font-weight:700;">{prob * 100:.1f}%</div>
  <div style="background:#374151;border-radius:6px;height:18px;margin-top:10px;">
    <div style="width:{prob * 100:.1f}%;background:{color};height:18px;border-radius:6px;"></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(_interpret(prob))

        with st.expander("Feature values used"):
            st.dataframe(row.T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
