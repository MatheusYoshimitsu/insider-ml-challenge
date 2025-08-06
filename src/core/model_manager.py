from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Class for managing machine learning models, including loading and reloading models.

    Attributes:
        model_path (Path): Path to the initial model file.
        model: Loaded model object.
    """

    def __init__(self, model_path: Path):
        """
        Initialize the ModelManager by loading the specified model.

        Args:
            model_path (Path): Path to the model file to load.
        """
        self.model_path = model_path
        self.model = self.load_model(model_path)

    def load_model(self, model_path: Path):
        """
        Load a model from the given file path.

        Args:
            model_path (Path): Path to the model file.

        Returns:
            object: Loaded model object.
        """
        logger.info(f"Loading model from {model_path}")
        return joblib.load(model_path)

    def reload_model(self, model_name: str):
        """
        Reload a model from the 'models' directory based on a given model name.

        Args:
            model_name (str): Name of the model file to load.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
        """
        models_dir = self.model_path.parent
        new_path = models_dir / model_name
        if not new_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found.")
        self.model = self.load_model(new_path)
        logger.info(f"Model reloaded: {model_name}")
