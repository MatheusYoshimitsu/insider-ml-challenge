from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

# Import so joblib can find feature_engineering when unpickling
from src.train import feature_engineering

# Globals
model = None
history = []


class PassengerData(BaseModel):
    """Data schema for prediction requests."""

    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str | None = None
    Embarked: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model at startup and cleanup at shutdown.
    """
    global model
    model_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "best_rf_pipeline.pkl"
    )

    if not model_path.exists():
        raise RuntimeError(f"Model file not found at {model_path}")

    # Load the pipeline (which uses feature_engineering)
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")

    yield  # Application runs

    print("API shutting down...")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict")
def predict(data: List[PassengerData]):
    """
    Predict survival for passengers.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert input list into a DataFrame
    df = pd.DataFrame([item.model_dump() for item in data])
    preds = model.predict(df).tolist()

    # Save predictions to history
    for passenger, pred in zip(data, preds):
        history.append({"input": passenger.model_dump(), "prediction": pred})

    return {"predictions": preds}


@app.post("/load")
def load_new_model(model_filename: str):
    """
    Load a new model dynamically from the models folder.
    """
    global model
    model_path = (
        Path(__file__).resolve().parent.parent / "models" / model_filename
    )

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    model = joblib.load(model_path)
    return {"status": "model loaded", "model": model_filename}


@app.get("/history")
def get_history():
    """
    Return the prediction call history.
    """
    return {"history": history}
