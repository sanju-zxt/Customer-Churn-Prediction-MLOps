from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.api import api_router

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "artifacts" / "model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        app.state.model = joblib.load(MODEL_PATH)
    else:
        app.state.model = None
    app.state.piper_voices = {}
    yield
    app.state.model = None


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using the trained scikit-learn pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(api_router)


class CustomerFeatures(BaseModel):
    tenure: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)
    support_calls: int = Field(ge=0)
    contract: str
    internet_service: str | None = None


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features = pd.DataFrame([customer.model_dump()])

    try:
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        return {
            "churn_prediction": prediction,
            "churn_label": (
                "Likely to churn" if prediction == 1 else "Unlikely to churn"
            ),
            "churn_probability": round(float(probabilities[1]), 4),
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
