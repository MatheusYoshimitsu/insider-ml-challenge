from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from pathlib import Path
import pandas as pd
from src.api.schemas import PassengerData
from src.core.model_manager import ModelManager
from src.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()

model_manager = ModelManager(
    Path(__file__).resolve().parents[2] / "models" / "best_rf_pipeline.pkl"
)

prediction_history = []


@router.get("/", include_in_schema=False)
def root():
    """
    Redirects root endpoint to the API docs.

    Returns:
        RedirectResponse: Redirect to /docs.
    """
    return RedirectResponse(url="/docs")


@router.get("/health")
def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Status message indicating the service is running.
    """
    return {"status": "ok"}


@router.get("/history")
def get_history():
    """
    Retrieve the prediction history.

    Returns:
        list: List of dictionaries containing input and prediction results.
    """
    return prediction_history


@router.post("/predict")
def predict(data: PassengerData):
    """
    Predict survival based on passenger data.

    Args:
        data (PassengerData): Input passenger data.

    Returns:
        dict: Prediction result with the survival class (0, 1).

    Raises:
        HTTPException: If prediction fails due to model error.
    """
    logger.info("Prediction request received")
    df = pd.DataFrame([data.model_dump()])
    try:
        prediction = int(model_manager.model.predict(df)[0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed")

    prediction_history.append(
        {"input": data.model_dump(), "prediction": prediction}
    )
    return {"prediction": prediction}


@router.post("/load")
def load_model(model_name: str):
    """
    Load a different model from the models directory.

    Args:
        model_name (str): Filename of the model to load.

    Returns:
        dict: Success message if model is loaded.

    Raises:
        HTTPException: If the specified model file is not found.
    """
    try:
        model_manager.reload_model(model_name)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": f"Model {model_name} loaded successfully"}
