
from pathlib import Path

import pandas as pd


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
]


def validate_data(file_path) -> pd.DataFrame:
    """Load and validate the customer churn dataset.

    Rejects missing values in numeric columns. Missing categorical
    feature values are allowed because the training pipeline can impute
    them. The churn target must never be missing.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Validate required columns.
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Report missing values before enforcing the validation rules.
    missing_counts = df[REQUIRED_COLUMNS].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        print("Missing values found:")
        print(missing_counts.to_string())

    # Numeric features must be valid numbers.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_numeric = df[NUMERIC_COLUMNS].isna().sum()
    invalid_numeric = invalid_numeric[invalid_numeric > 0]

    if not invalid_numeric.empty:
        raise ValueError(
            "missing values in numeric columns: "
            f"{invalid_numeric.to_dict()}"
        )

    # The target cannot be missing.
    if df["churn"].isna().any():
        raise ValueError("missing values in target column: churn")

    # Categorical fields can contain missing values, which the model
    # pipeline handles using SimpleImputer.
    print("\nData validation passed!")
    print(f"Total rows: {len(df)}")
    print("All required columns are present.")
    print("Numeric features and target contain no missing values.")

    return df
