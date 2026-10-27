import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from validation import validate_data


SEED = 42
MINIMUM_AUC = 0.60
TEST_SIZE = 0.20
MAX_ITER = 1000

NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "support_calls",
]

CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
]

TARGET = "churn"


def build_model():
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=MAX_ITER,
                    random_state=SEED,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check-quality-gate",
        action="store_true",
    )

    args = parser.parse_args()

    # Load and validate the dataset
    df = validate_data()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # Create artifacts directory
    Path("artifacts").mkdir(exist_ok=True)

    # Set up MLflow experiment
    mlflow.set_experiment("Customer Churn Prediction")

    with mlflow.start_run() as run:
        # Build and train the model
        model = build_model()
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        # Log parameters to MLflow
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "random_seed": SEED,
                "test_size": TEST_SIZE,
                "max_iter": MAX_ITER,
                "dataset_rows": len(df),
            }
        )

        # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "accuracy": float(accuracy),
                "auc": float(auc),
            }
        )

        # Save model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        # Save model locally
        joblib.dump(model, "artifacts/model.joblib")

        # Prepare metrics report
        metrics = {
            "accuracy": float(accuracy),
            "auc": float(auc),
            "minimum_auc": MINIMUM_AUC,
            "run_id": run.info.run_id,
            "rows": len(df),
        }

        with open("artifacts/metrics.json", "w") as file:
            json.dump(metrics, file, indent=4)

        # Display results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")

        # Check model quality
        if args.check_quality_gate:
            if auc < MINIMUM_AUC:
                raise RuntimeError(
                    f"Quality gate failed: ROC-AUC {auc:.4f} "
                    f"is below {MINIMUM_AUC:.2f}"
                )

            print("Quality gate passed.")
        else:
            print("Quality gate check was not requested.")

        print("Model saved to artifacts/model.joblib")
        print("Metrics saved to artifacts/metrics.json")


if __name__ == "__main__":
    main()