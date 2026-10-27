
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "customers.csv"

REQUIRED_COLUMNS = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract",
    "internet_service",
    "churn",
]

NUMERIC_COLUMNS = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "churn",
]

CATEGORICAL_COLUMNS = [
    "contract",
    "internet_service",
]


def validate_data(file_path=DEFAULT_DATA_PATH):
    """Load and validate a customer churn dataset."""

    file_path = Path(file_path)

    if not file_path.is_absolute():
        file_path = PROJECT_DIR / file_path

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(
            f"Unable to read dataset: {exc}"
        ) from exc

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if df[column].isna().any():
            raise ValueError(
                f"missing values or invalid numeric values "
                f"in '{column}'."
            )

        if not np.isfinite(df[column].to_numpy()).all():
            raise ValueError(
                f"Non-finite numeric values in '{column}'."
            )

    if not df["churn"].isin([0, 1]).all():
        raise ValueError(
            "The 'churn' column must contain only 0 or 1."
        )

    if df["churn"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both churn classes: 0 and 1."
        )

    if df["contract"].isna().any():
        raise ValueError(
            "Missing values detected in 'contract'."
        )

    for column in CATEGORICAL_COLUMNS:
        values = df[column].dropna()

        if not values.map(
            lambda value: (
                isinstance(value, str)
                and bool(value.strip())
            )
        ).all():
            raise ValueError(
                f"Invalid categorical values in '{column}'."
            )

    return df


if __name__ == "__main__":
    dataset = validate_data()

    print("Data validation successful.")
    print(f"Dataset shape: {dataset.shape}")
    print("Missing values by column:")
    print(dataset[REQUIRED_COLUMNS].isna().sum())
    print(f"Churn rate: {dataset['churn'].mean():.2%}")
