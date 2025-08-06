import joblib
from pathlib import Path
from src.core.model_manager import ModelManager


def dummy_model(x):
    return x


def test_model_load_and_reload(tmp_path):
    model_file = tmp_path / "dummy.pkl"
    joblib.dump(dummy_model, model_file)

    manager = ModelManager(model_file)
    assert manager.model is not None

    # Create a second model
    second_model_file = tmp_path / "second.pkl"
    joblib.dump(dummy_model, second_model_file)

    manager.reload_model("second.pkl")
    assert manager.model is not None


def test_reload_model_not_found(tmp_path):
    model_file = tmp_path / "dummy.pkl"
    joblib.dump(dummy_model, model_file)

    manager = ModelManager(model_file)

    try:
        manager.reload_model("nonexistent.pkl")
    except FileNotFoundError as e:
        assert "not found" in str(e).lower()
