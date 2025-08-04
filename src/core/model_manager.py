from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = self.load_model(model_path)

    def load_model(self, model_path: Path):
        logger.info(f"Loading model from {model_path}")
        return joblib.load(model_path)

    def reload_model(self, model_name: str):
        models_dir = self.model_path.parent
        new_path = models_dir / model_name
        if not new_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found.")
        self.model = self.load_model(new_path)
        logger.info(f"Model reloaded: {model_name}")
