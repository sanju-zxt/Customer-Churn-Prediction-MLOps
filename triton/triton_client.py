# =============================================================================
# Triton Inference Server - HTTP Client Script
# =============================================================================
# Usage:
#   python triton_client.py --url http://localhost:8000/v2/models/churn_v1/infer
# =============================================================================
import argparse
import json
import time

import httpx


def create_churn_request(
    tenure: int,
    monthly_charges: float,
    support_calls: int,
    contract: str,
    internet_service: str | None = None,
) -> dict:
    """Create a churn prediction request for Triton Inference Server."""
    # Map string categorical values to numeric encodings
    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    }
    internet_map = {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2,
    }

    inputs = [
        {
            "name": "tenure_input",
            "shape": [1],
            "datatype": "INT32",
            "data": [tenure],
        },
        {
            "name": "monthly_charges_input",
            "shape": [1],
            "datatype": "FP32",
            "data": [monthly_charges],
        },
        {
            "name": "support_calls_input",
            "shape": [1],
            "datatype": "INT32",
            "data": [support_calls],
        },
        {
            "name": "contract_input",
            "shape": [1],
            "datatype": "FP32",
            "data": [float(contract_map.get(contract, 0))],
        },
        {
            "name": "internet_service_input",
            "shape": [1],
            "datatype": "FP32",
            "data": [float(internet_map.get(internet_service, 0)) if internet_service else 0.0],
        },
    ]

    return {
        "inputs": inputs,
        "outputs": [
            {
                "name": "churn_prediction",
                "parameters": {"binary_data": False},
            },
            {
                "name": "churn_probability",
                "parameters": {"binary_data": False},
            },
        ],
    }


def predict_churn(triton_url: str, customer_data: dict) -> dict:
    """Send prediction request to Triton Inference Server."""
    request_body = create_churn_request(**customer_data)

    # Health check
    health_url = triton_url.replace("/infer", "").rstrip("/")
    try:
        health_resp = httpx.get(
            f"{health_url}/v2/health/ready",
            params={"model_name": "churn_v1"},
            timeout=5,
        )
        health_resp.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"Triton server not ready: {e}") from e

    # Inference request
    start_time = time.time()
    response = httpx.post(
        f"{triton_url}",
        json=request_body,
        timeout=30,
        headers={"Content-Type": "application/json"},
    )
    latency_ms = (time.time() - start_time) * 1000

    response.raise_for_status()
    result = response.json()

    # Parse response
    prediction = int(result["outputs"][0]["data"][0])
    probability = float(result["outputs"][1]["data"][0])

    return {
        "churn_prediction": prediction,
        "churn_label": "Likely to churn" if prediction == 1 else "Unlikely to churn",
        "churn_probability": round(probability, 4),
        "latency_ms": round(latency_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Triton Inference Client for Churn Prediction")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v2/models/churn_prediction_model/infer",
        help="Triton inference endpoint URL",
    )
    parser.add_argument("--tenure", type=int, default=12, help="Tenure in months")
    parser.add_argument("--monthly-charges", type=float, default=70.50, help="Monthly charges")
    parser.add_argument("--support-calls", type=int, default=2, help="Number of support calls")
    parser.add_argument("--contract", default="Month-to-month", help="Contract type")
    parser.add_argument("--internet-service", default="Fiber optic", help="Internet service type")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    customer_data = {
        "tenure": args.tenure,
        "monthly_charges": args.monthly_charges,
        "support_calls": args.support_calls,
        "contract": args.contract,
        "internet_service": args.internet_service,
    }

    try:
        result = predict_churn(args.url, customer_data)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Prediction: {result['churn_label']}")
            print(f"Probability: {result['churn_probability']}")
            print(f"Latency: {result['latency_ms']}ms")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()