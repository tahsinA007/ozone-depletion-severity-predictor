"""
train_model.py
---------------
Reproduces the exact pipeline from Ozone_O3__layer_depletion_prediction.ipynb:
  1. Load dataset
  2. Impute missing values (mean for numeric, most_frequent for categorical)
  3. Label-encode the 4 categorical columns (separate encoder per column)
  4. Train/test split (80/20, stratified, random_state=42)
  5. RobustScaler fit ONLY on training numeric columns
  6. Map target to {Low Depletion:1, Moderate Depletion:2, Severe Depletion:3}
  7. Train the selected final model: Softmax Regression
     (LogisticRegression, solver='lbfgs', max_iter=1000, random_state=42)
  8. Save every artifact the Streamlit app needs: model, scaler, encoders,
     feature order, and evaluation metrics (real numbers, not the mockup's
     placeholder numbers).

Run this once (`python train_model.py`) before launching the Streamlit app.
It creates an `artifacts/` folder with everything app.py loads.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

DATA_PATH = "ozone_layer_depletion_dataset.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RISK_MAP = {"Low Depletion": 1, "Moderate Depletion": 2, "Severe Depletion": 3}
LABEL_MAP = {v: k.replace(" Depletion", "") for k, v in RISK_MAP.items()}  # 1->Low ...

NUM_COLS = [
    "Stratospheric Temperature (Celsius)",
    "Chlorine Monoxide Concentration (Parts Per Billion)",
    "Bromine Monoxide Concentration (Parts Per Trillion)",
    "CFC-11 Concentration (Parts Per Trillion)",
    "CFC-12 Concentration (Parts Per Trillion)",
    "Nitrous Oxide Concentration (Parts Per Billion)",
    "UV Index",
    "Polar Stratospheric Cloud Index",
    "Wind Speed (Kilometers Per Hour)",
    "Latitude (Degrees)",
    "Methane Concentration (Parts Per Billion)",
]
CAT_COLS = ["Season", "Hemisphere", "Region Type", "Monitoring Station Type"]
TARGET_COL = "Ozone Depletion Level"


def main():
    # 1. Load
    df = pd.read_csv(DATA_PATH)

    # 2. Missing value treatment (mean / most_frequent, same as notebook)
    num_col = df.select_dtypes(include="number").columns
    cat_col = df.select_dtypes(include="object").columns
    num_si = SimpleImputer(strategy="mean")
    df[num_col] = num_si.fit_transform(df[num_col])
    cat_si = SimpleImputer(strategy="most_frequent")
    df[cat_col] = cat_si.fit_transform(df[cat_col])

    # 3. Label encode categoricals — ONE encoder per column (fixes the bug in
    #    the notebook where a single reused LabelEncoder overwrote itself)
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col + "_Encoded"] = le.fit_transform(df[col])
        encoders[col] = le

    # 4. Feature/target separation — same column drop/order as the notebook
    X = df.drop([TARGET_COL] + CAT_COLS, axis=1)
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 6. Scale numeric columns (fit on train only)
    scaler = RobustScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

    # 7. Map target to {1,2,3}
    y_train = y_train.map(RISK_MAP)
    y_test = y_test.map(RISK_MAP)

    # 8a. Baseline Perceptron (for the Model Insights comparison table)
    perceptron = Perceptron(max_iter=1000, eta0=1.0, random_state=42)
    perceptron.fit(X_train, y_train)
    y_pred_p = perceptron.predict(X_test)

    # 8b. Final selected model: Softmax Regression
    softmax_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    softmax_model.fit(X_train, y_train)
    y_pred_s = softmax_model.predict(X_test)

    # 9. Metrics (real numbers computed from this run, not mockup placeholders)
    report_p = classification_report(y_test, y_pred_p, output_dict=True)
    report_s = classification_report(y_test, y_pred_s, output_dict=True)
    cm_softmax = confusion_matrix(y_test, y_pred_s).tolist()
    cm_perceptron = confusion_matrix(y_test, y_pred_p).tolist()

    metrics = {
        "perceptron": {
            "test_accuracy": accuracy_score(y_test, y_pred_p),
            "train_accuracy": accuracy_score(y_train, perceptron.predict(X_train)),
            "precision_macro": report_p["macro avg"]["precision"],
            "recall_macro": report_p["macro avg"]["recall"],
            "f1_macro": report_p["macro avg"]["f1-score"],
            "confusion_matrix": cm_perceptron,
        },
        "softmax": {
            "test_accuracy": accuracy_score(y_test, y_pred_s),
            "train_accuracy": accuracy_score(y_train, softmax_model.predict(X_train)),
            "precision_macro": report_s["macro avg"]["precision"],
            "recall_macro": report_s["macro avg"]["recall"],
            "f1_macro": report_s["macro avg"]["f1-score"],
            "confusion_matrix": cm_softmax,
        },
        "class_order": [1, 2, 3],  # Low, Moderate, Severe
        "class_labels": ["Low", "Moderate", "Severe"],
    }

    # Feature importance: mean |coefficient| across the 3 one-vs-rest rows
    coef = np.abs(softmax_model.coef_).mean(axis=0)
    importance = sorted(
        zip(feature_names, coef.tolist()), key=lambda t: t[1], reverse=True
    )

    # 10. Save everything the app needs
    joblib.dump(softmax_model, f"{ARTIFACT_DIR}/softmax_model.joblib")
    joblib.dump(scaler, f"{ARTIFACT_DIR}/scaler.joblib")
    joblib.dump(encoders, f"{ARTIFACT_DIR}/label_encoders.joblib")
    joblib.dump(feature_names, f"{ARTIFACT_DIR}/feature_names.joblib")
    with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{ARTIFACT_DIR}/feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    print("Training complete. Artifacts saved to:", ARTIFACT_DIR)
    print(f"Softmax test accuracy: {metrics['softmax']['test_accuracy']:.2%}")
    print(f"Perceptron test accuracy: {metrics['perceptron']['test_accuracy']:.2%}")
    print("\nFeature importance (|coef| avg, descending):")
    for name, val in importance:
        print(f"  {name}: {val:.3f}")


if __name__ == "__main__":
    main()