"""Train and export a lightweight model from scikit-learn's WDBC dataset.

The model is for educational demonstration only. It is not a clinical device.
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "model.json"


def main() -> None:
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    evaluation_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
    )
    evaluation_model.fit(X_train, y_train)
    pred = evaluation_model.predict(X_test)
    # target 0 = malignant; target 1 = benign. Sensitivity here refers to malignant class recall.
    malignant_recall = recall_score(y_test, pred, pos_label=0)
    metrics = {
        "holdout_size": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "roc_auc_malignant_score": round(
            float(roc_auc_score(1 - y_test, -evaluation_model.decision_function(X_test))), 4
        ),
        "malignant_recall": round(float(malignant_recall), 4),
        "benign_precision": round(float(precision_score(y_test, pred, pos_label=1)), 4),
        "confusion_matrix_labels_malignant_benign": confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).tolist(),
        "split": "stratified 80/20 holdout; random_state=42",
    }

    final_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
    )
    final_model.fit(X, y)
    scaler = final_model.named_steps["standardscaler"]
    classifier = final_model.named_steps["logisticregression"]

    payload = {
        "model_name": "WDBC Logistic Regression",
        "model_version": "1.0.0",
        "dataset": {
            "name": "Wisconsin Diagnostic Breast Cancer (WDBC)",
            "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
            "scikit_learn_reference": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
            "sample_count": int(X.shape[0]),
            "feature_count": int(X.shape[1]),
            "class_mapping": {"0": "malignant", "1": "benign"},
            "feature_names": list(dataset.feature_names),
        },
        "training": {
            "algorithm": "StandardScaler + LogisticRegression",
            "max_iter": 2000,
            "final_fit_samples": int(X.shape[0]),
            "holdout_metrics": metrics,
            "metrics_are_from_holdout_before_final_refit": True,
        },
        "inference": {
            "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(),
            "coefficient_benign": classifier.coef_[0].tolist(),
            "intercept_benign": float(classifier.intercept_[0]),
            "probability_class": "benign",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_artifact": str(OUT), "holdout_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
