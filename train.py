
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

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

from validation import validate_data


PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_DIR / "data" / "customers.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

TARGET_COLUMN = "churn"

NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "support_calls",
]

CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
]

MINIMUM_AUC = 0.60
RANDOM_STATE = 42


def build_model():
    """Create the preprocessing and classification pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing",
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

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the customer churn model."
    )

    parser.add_argument(
        "--check-quality-gate",
        action="store_true",
        help="Fail when ROC-AUC is below the minimum threshold.",
    )

    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = validate_data(DATA_PATH)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model()

    tracking_dir = PROJECT_DIR / "mlruns"
    tracking_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("customer-churn-prediction")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        auc_score = float(
            roc_auc_score(y_test, probabilities)
        )

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "auc": auc_score,
            "roc_auc": auc_score,
            "minimum_auc": MINIMUM_AUC,
        }

        mlflow.log_param(
            "model_type",
            "LogisticRegression",
        )
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("minimum_auc", MINIMUM_AUC)
        mlflow.log_metrics(metrics)

        joblib.dump(model, MODEL_PATH)

        with open(
            METRICS_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metrics, file, indent=4)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        print("\nModel training completed.")
        print(f"Training rows: {len(X_train)}")
        print(f"Testing rows: {len(X_test)}")
        print("\nEvaluation metrics:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"Metrics saved to: {METRICS_PATH}")

        if args.check_quality_gate:
            if auc_score < MINIMUM_AUC:
                raise SystemExit(
                    f"QUALITY GATE FAILED: ROC-AUC {auc_score:.4f} "
                    f"is below the required {MINIMUM_AUC:.2f}."
                )

            print(
                f"\nQUALITY GATE PASSED: ROC-AUC {auc_score:.4f} "
                f">= {MINIMUM_AUC:.2f}"
            )


if __name__ == "__main__":
    main()
