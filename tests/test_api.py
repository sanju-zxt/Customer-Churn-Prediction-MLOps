
from fastapi.testclient import TestClient

from app import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Customer Churn Prediction API" in response.json()["message"]


def test_predict_endpoint():
    payload = {
        "tenure": 12,
        "monthly_charges": 79.99,
        "support_calls": 3,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert result["churn_prediction"] in [0, 1]
    assert result["churn_label"] in ["Likely to churn", "Unlikely to churn"]
    assert 0 <= result["churn_probability"] <= 1


def test_predict_rejects_invalid_input():
    payload = {
        "tenure": -1,
        "monthly_charges": 79.99,
        "support_calls": 3,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
