from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
ROWS = 2000


def main():
    rng = np.random.default_rng(SEED)

    tenure = rng.integers(1, 73, size=ROWS)

    monthly_charges = np.round(
        rng.normal(70, 20, size=ROWS).clip(20, 130), 2
    )

    support_calls = rng.poisson(1.5, size=ROWS)

    contract = rng.choice(
        ["Monthly", "One Year", "Two Year"],
        size=ROWS,
        p=[0.55, 0.25, 0.20],
    )

    internet_service = rng.choice(
        ["DSL", "Fiber", "None"],
        size=ROWS,
        p=[0.35, 0.50, 0.15],
    )

    # A synthetic churn tendency, not a real-world rule.
    score = (
        -1.0
        - 0.035 * tenure
        + 0.018 * (monthly_charges - 70)
        + 0.38 * support_calls
        + 1.1 * (contract == "Monthly")
        - 0.7 * (contract == "Two Year")
        + 0.45 * (internet_service == "Fiber")
    )

    probability = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, probability)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "support_calls": support_calls,
        "contract": contract,
        "internet_service": internet_service,
        "churn": churn,
    })

    output = Path("data/customers.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output, index=False)

    print(f"Dataset created: {output}")
    print(f"Rows and columns: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(df.head())


if __name__ == "__main__":
    main()