# Coronary Artery Disease Risk Estimator

> Feature-selection-driven binary classifier for coronary artery disease, trained on the UCI Cleveland heart disease dataset and deployed as a Streamlit app.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/HF%20Spaces-Live%20Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/tejaswirajgit/coronary-artery-disease)

**Live demo:** <https://huggingface.co/spaces/tejaswirajgit/coronary-artery-disease>

![App screenshot](assets/screenshots/app.png)

> [!WARNING]
> **Educational project — not medical advice.** This model is trained on 303 records from a 1988 UCI dataset and has not been clinically validated. Do not use for diagnosis, treatment, or any medical decision.

---

## Problem

Coronary artery disease (CAD) is typically confirmed by angiography — an invasive, expensive procedure. This project asks: given routinely-collected patient data (demographics, stress-test results, ECG findings), how accurately can we predict CAD before angiography, and which features actually matter? The answer informs which non-invasive measurements clinicians should prioritise.

## Dataset

- **Source:** [UCI Cleveland heart disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Size:** 303 patients, 14 columns
- **Target:** `ca_disease` (0 = no disease, 1–4 = increasing severity → binarised to 0/1)
- **Class balance:** roughly 54% / 46% — no rebalancing required.
- **Missing values:** 6 total (`?` sentinels in `num_major_vessels` and `thallium_scint`) → median imputation.

## Methodology

1. **Cleaning** — coerce `?` to NaN, median-impute the 6 missing values, remap `thallium_scint` codes `{3:0, 7:1, 6:2}` so they are ordered by severity, binarise the target.
2. **EDA** — distributions, correlations, violin plots for continuous features, bar charts for categoricals.
3. **Feature selection** — four methods compared, keeping features that scored highly in at least two:
   - statsmodels Logit coefficient significance
   - `mlxtend` exhaustive 1-feature search
   - `SelectKBest` (`f_classif` for continuous, `chi2` for categorical)
   - Decision Tree and Random Forest feature importances
4. **Modelling** — scaled Logistic Regression (deployed) and a Multi-Layer Perceptron (compared). Identical `random_state=0` split as the notebook for reproducibility.

## Results

Final 8-feature scaled Logistic Regression vs the 13-feature unscaled baseline (test set: 76 patients):

| Metric    | Baseline (13 features) | Final (8 features, scaled) |
| --------- | ---------------------: | -------------------------: |
| Accuracy  |                 0.8158 |                 **0.8289** |
| Precision |                 0.8929 |                 **0.8710** |
| Recall    |                 0.6944 |                 **0.7500** |
| ROC-AUC   |                 0.8785 |                 **0.8861** |

The final model uses fewer features, scales every feature so convergence is dramatically faster, and improves on the baseline across accuracy, recall, and ROC-AUC — at a small cost in precision.

## Selected features

| Feature | Why kept |
|---|---|
| `num_major_vessels` | Strongest discriminator across every selection method; largest scaled coefficient. |
| `thallium_scint` | High chi² + univariate accuracy; clinically the closest non-invasive proxy for CAD. |
| `exer_ind_angina` | Strong binary predictor; small confidence interval in univariate test. |
| `chest_pain_type` | High chi² score; the four-level ordinal captures angina character well. |
| `st_slope` | Significant chi²; complements ST depression to characterise the stress-test response. |
| `st_depression` | Significant logistic coefficient; differs sharply between CAD and non-CAD distributions. |
| `max_heart_rate` | Significant logistic coefficient and high SelectKBest score for continuous variables. |
| `sex` | Smaller effect but consistently above the noise floor across methods. |

Features dropped (`age`, `rest_blood_press`, `cholesterol`, `fasting_blood_sugar`, `rest_ecg`) either failed every test or had their signal absorbed by a correlated retained feature (e.g. age ↔ max_heart_rate).

## Run locally

```bash
git clone https://github.com/tejaswirajgit/Coronary-Artery-Disease-Feature-Selection.git
cd Coronary-Artery-Disease-Feature-Selection
pip install -r requirements.txt
python -m src.train         # writes models/model.pkl, models/scaler.pkl, models/metrics.json
streamlit run app.py
```

## Project structure

```text
.
├── app.py                       # Streamlit app (HF Spaces entry point)
├── data/
│   └── processed_cleveland.data
├── models/
│   ├── model.pkl                # Trained Logistic Regression
│   ├── scaler.pkl               # StandardScaler fit on train split
│   └── metrics.json             # Test-set metrics + coefficients
├── notebooks/
│   └── coronary-artery-disease-feature-selection.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # Cleaning + feature selection
│   └── train.py                 # Reproducible training pipeline
├── .github/workflows/train.yml  # CI: retrains on every push to main
├── requirements.txt
├── LICENSE
└── README.md
```

## Tech stack

- **Modelling:** scikit-learn (LogisticRegression, StandardScaler, train_test_split)
- **Data:** pandas, numpy
- **App:** Streamlit
- **Persistence:** joblib
- **Plotting (sidebar):** matplotlib
- **Notebook-only extras:** statsmodels, mlxtend, seaborn, Graphviz (for tree visualisations)
- **CI:** GitHub Actions

## Limitations

- Only 303 patients; ROC-AUC of 0.89 on 76 test samples carries non-trivial uncertainty.
- Single-centre, single-era data (Cleveland Clinic, late 1980s) — no guarantee of external validity for other populations or modern clinical protocols.
- The model exposes correlations, not causation.
- Class labels were binarised — severity information (originally 0–4) is discarded.
- **Not validated for clinical use under any circumstances.**

## License

[MIT](LICENSE). The Cleveland dataset is provided by the UCI Machine Learning Repository under their own terms.
