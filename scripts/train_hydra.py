import sys
from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from validation import validate_data


NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "support_calls",
]

CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
]

TARGET_COLUMN = "churn"


def build_model(cfg):
    """Create the preprocessing and classification pipeline from config."""
    hp = cfg.model.hyperparameters

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cfg.data.preprocessing.numeric_imputer)),
            ("scaler", StandardScaler()),
        ]
    )

    cat_imputer_cfg = cfg.data.preprocessing.categorical_imputer
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=cat_imputer_cfg.strategy,
                    fill_value=cat_imputer_cfg.fill_value,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier_kwargs = {
        "max_iter": hp.max_iter,
        "random_state": hp.random_state,
        "C": hp.C,
        "solver": hp.solver,
        "penalty": hp.penalty,
    }

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(**classifier_kwargs),
            ),
        ]
    )


def setup_mlflow(cfg):
    """Configure MLflow tracking from config."""
    mlflow_cfg = cfg.experiment.mlflow
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(mlflow_cfg.experiment_name)


@hydra.main(
    config_path="../conf",
    config_name="config",
    version_base="1.3",
)
def main(cfg: OmegaConf) -> None:
    print(f"\nHydra config:\n{OmegaConf.to_yaml(cfg)}")

    setup_mlflow(cfg)

    ckpt_cfg = cfg.training.checkpointing
    artifacts_dir = Path(ckpt_cfg.model_path).parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = validate_data(cfg.data.path)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y,
    )

    model = build_model(cfg)

    mlflow_cfg = cfg.experiment.mlflow
    run_name = mlflow_cfg.run_name.format(
        model_name=cfg.model.name,
        data_name=cfg.data.name,
    ) if "{model_name" in (mlflow_cfg.run_name or "") else mlflow_cfg.run_name

    with mlflow.start_run(run_name=run_name):
        if mlflow_cfg.log_params:
            mlflow.log_params({
                "model_type": cfg.model.hyperparameters.model_type,
                "max_iter": cfg.model.hyperparameters.max_iter,
                "random_state": cfg.model.hyperparameters.random_state,
                "C": cfg.model.hyperparameters.C,
                "solver": cfg.model.hyperparameters.solver,
                "penalty": cfg.model.hyperparameters.penalty,
                "test_size": cfg.training.test_size,
                "minimum_auc": cfg.training.quality_gate.minimum_auc,
            })

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        auc_score = float(roc_auc_score(y_test, probabilities))

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "auc": auc_score,
            "roc_auc": auc_score,
        }

        if mlflow_cfg.log_metrics:
            mlflow.log_metrics(metrics)

        if ckpt_cfg.save_model:
            joblib.dump(model, ckpt_cfg.model_path)

        if ckpt_cfg.save_metrics:
            metrics_path = Path(ckpt_cfg.metrics_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                import json
                json.dump(metrics, f, indent=4)

        if mlflow_cfg.log_artifacts:
            if ckpt_cfg.save_model:
                mlflow.log_artifact(ckpt_cfg.model_path)
            if ckpt_cfg.save_metrics:
                mlflow.log_artifact(ckpt_cfg.metrics_path)
            mlflow.log_artifact(str(Path(cfg.config_dir) / "config.yaml"))
            mlflow.log_artifact(str(Path(cfg.config_dir) / "model" / "logistic_regression.yaml"))

        if mlflow_cfg.log_model:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
            )

        print("\nModel training completed.")
        print(f"Training rows: {len(X_train)}")
        print(f"Testing rows: {len(X_test)}")
        print("\nEvaluation metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print(f"\nModel saved to: {ckpt_cfg.model_path}")
        print(f"Metrics saved to: {ckpt_cfg.metrics_path}")

        min_auc = cfg.training.quality_gate.minimum_auc
        if auc_score < min_auc:
            raise SystemExit(
                f"QUALITY GATE FAILED: ROC-AUC {auc_score:.4f} "
                f"is below the required {min_auc:.2f}."
            )

        print(f"\nQUALITY GATE PASSED: ROC-AUC {auc_score:.4f} >= {min_auc:.2f}")


if __name__ == "__main__":
    main()
