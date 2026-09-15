# Heart Disease Prediction — Task 4

Predicting the presence of heart disease from structured clinical data using
classification algorithms, evaluated with a strict no-leakage methodology.

## Overview

This project trains and compares four classifiers — **Logistic Regression,
SVM, Random Forest, and XGBoost** — on the UCI Cleveland Heart Disease
dataset to predict `target`: `0` = no disease, `1` = disease. The final
model is deployed as an interactive **Streamlit** app.

## Dataset

`heart.csv` (included via `data/`) is a Kaggle mirror of the UCI Cleveland
Heart Disease dataset. As distributed, it contains **1025 rows, but 723 of
those are exact duplicates** of the original **302 unique patient records**.
This project removes the duplicates *before* the train/test split — training
or evaluating on the un-deduplicated file causes data leakage (the same
patient can appear in both splits) and is why some public notebooks report
inflated 95–100% accuracy on this dataset.

Two additional columns use out-of-range integer codes for missing data:
`ca == 4` and `thal == 0`. These are treated as missing values and imputed
inside the training pipeline (never using test data).

## Methodology highlights

- Stratified 80/20 train/test split (test set touched exactly once)
- `ColumnTransformer` + `Pipeline` for all preprocessing (imputation,
  scaling, one-hot encoding) — fit only on training folds, never leaked
- 5-fold stratified cross-validation for all model comparisons
- `RandomizedSearchCV` hyperparameter tuning per model
- In-pipeline feature selection experiment (`SelectKBest`) — tested, not
  adopted, since it didn't improve CV performance
- Ensemble (soft voting) tested, not adopted
- Robustness check across 5 random seeds to report stability, not to
  cherry-pick a favorable split
- Final model selected by CV ROC-AUC, stability, and interpretability —
  not by test accuracy alone

## Results

| Model | CV Accuracy | Test Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression (final)** | 0.846 ± 0.072 | **0.852** | 0.875 | 0.848 | 0.862 | 0.906 |
| SVM | 0.842 ± 0.078 | 0.869 | 0.879 | 0.879 | 0.879 | 0.898 |
| Random Forest | 0.825 ± 0.078 | 0.820 | 0.844 | 0.818 | 0.831 | 0.894 |
| XGBoost | 0.830 ± 0.055 | 0.803 | 0.862 | 0.758 | 0.806 | 0.870 |

**Did we hit 90%+ accuracy?** No. After honest deduplication, only 302
unique patients remain, and cross-validated / test performance both land in
the mid-to-high 80s — consistent with published, leakage-free results on
this exact dataset. Full reasoning is in the notebook's final section.

## Project structure

```
.
├── Heart_Disease_Prediction.ipynb   # full analysis, executed with real outputs
├── data/
│   └── heart.csv                    # raw dataset
├── outputs/
│   ├── heart_disease_model.pkl      # saved pipeline (preprocessing + model)
│   └── *.png                        # EDA and evaluation plots
├── app.py                           # Streamlit prediction app
├── requirements.txt
└── README.md
```

## Running locally

```bash
pip install -r requirements.txt

# Run the notebook
jupyter notebook Heart_Disease_Prediction.ipynb

# Run the app (expects heart_disease_model.pkl next to app.py,
# or update MODEL_PATH in app.py to point at outputs/heart_disease_model.pkl)
streamlit run app.py
```

## Disclaimer

This project is for educational and research purposes only and is **not**
a medical diagnostic tool.
