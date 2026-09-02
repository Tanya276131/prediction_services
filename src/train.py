"""
Trains multiple churn-prediction model variants, tracks every run in
MLflow (params, metrics, the model artifact), and promotes the best
run's model into models/production_model.joblib for the API to serve.

Run: python -m src.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.data.preprocess import TARGET_COL, run_pipeline

MLFLOW_EXPERIMENT = "churn-prediction"
MODELS_DIR = Path("models")


def get_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
        )
    return models


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main():
    df = run_pipeline()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    best_score = -1.0
    best_run_info = None

    for name, model in get_models().items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)

            mlflow.log_param("model_type", name)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"[{name}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_run_info = {
                    "run_id": run.info.run_id,
                    "model_name": name,
                    "metrics": metrics,
                }
                best_model = model

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "production_model.joblib")
    with open(MODELS_DIR / "production_model_info.json", "w") as f:
        json.dump(best_run_info, f, indent=2)

    print(f"\nPromoted '{best_run_info['model_name']}' to production "
          f"(roc_auc={best_score:.4f}, run_id={best_run_info['run_id']})")


if __name__ == "__main__":
    main()
