from __future__ import annotations
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC

from src.utils import feature_engineering, get_logger

logger = get_logger(__name__)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load Titanic dataset from a CSV file.

    Args:
        dataset_path (Path): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the dataset_path does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    logger.info(f"Loading dataset from {dataset_path}...")
    return pd.read_csv(dataset_path)


def build_preprocessor(
    categorical_cols: list[str], numerical_cols: list[str]
) -> ColumnTransformer:
    """Build a preprocessor for categorical and numerical columns.

    Args:
        categorical_cols (list[str]): List of categorical column names.
        numerical_cols (list[str]): List of numerical column names.

    Returns:
        ColumnTransformer: Preprocessing transformer for features.
    """
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
            ("numerical", StandardScaler(), numerical_cols),
        ]
    )


def build_pipelines(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Create machine learning pipelines for Random Forest and SVC models.

    Args:
        preprocessor (ColumnTransformer): Preprocessor for data transformation.

    Returns:
        dict[str, Pipeline]: Dictionary of pipelines keyed by model name.
    """
    rf_pipeline = Pipeline(
        [
            ("feature_eng", FunctionTransformer(feature_engineering)),
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=29)),
        ]
    )

    svc_pipeline = Pipeline(
        [
            ("feature_eng", FunctionTransformer(feature_engineering)),
            ("preprocessor", preprocessor),
            ("model", SVC(probability=True, random_state=29)),
        ]
    )

    return {"RandomForest": rf_pipeline, "SVC": svc_pipeline}


def evaluate_model(
    model: GridSearchCV, model_name: str, X_val: pd.DataFrame, y_val: pd.Series
) -> None:
    """Evaluate a trained model using validation data.

    Args:
        model (GridSearchCV): Trained GridSearchCV model.
        model_name (str): Name of the model for logging purposes.
        X_val (pd.DataFrame): Validation feature set.
        y_val (pd.Series): Validation target values.
    """
    logger.info(f"Evaluating {model_name} model...")
    logger.info(f"Best parameters: {model.best_params_}")

    y_pred = model.predict(X_val)
    logger.info(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    logger.info(f"F1 Score: {f1_score(y_val, y_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_val, y_pred):.4f}")
    logger.info(f"Recall: {recall_score(y_val, y_pred):.4f}")
    logger.info(
        f"Classification Report:\n{classification_report(y_val, y_pred)}"
    )


def perform_grid_search(
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict | list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Perform hyperparameter tuning using GridSearchCV.

    Args:
        model_name (str): Name of the model for logging purposes.
        pipeline (Pipeline): Machine learning pipeline.
        param_grid (dict | list): Hyperparameter grid.
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target values.

    Returns:
        GridSearchCV: Fitted GridSearchCV instance.
    """
    logger.info(f"Starting GridSearchCV for {model_name}...")
    grid = GridSearchCV(
        pipeline, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1
    )
    return grid.fit(X_train, y_train)


def save_model(model: Pipeline, path: Path) -> None:
    """Save a trained model to disk.

    Args:
        model (Pipeline): Trained model pipeline.
        path (Path): File path to save the model.
    """
    joblib.dump(model, path)
    logger.info(f"Model saved: {path}")


def main() -> None:
    """Main function to train and save Titanic survival prediction models."""
    dataset_path = Path(__file__).resolve().parents[1] / "dataset" / "train.csv"
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(dataset_path)
    y = train_df["Survived"]
    X_train, X_val, y_train, y_val = train_test_split(
        train_df, y, test_size=0.2, random_state=29, stratify=y
    )

    categorical_cols = [
        "Pclass",
        "Sex",
        "Embarked",
        "cabin_categories",
        "cabin_multiple",
        "name_title",
    ]
    numerical_cols = ["Age", "SibSp", "Parch", "norm_fare"]

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    pipelines = build_pipelines(preprocessor)

    param_grid_rf = {
        "model__n_estimators": [100, 200, 400, 500],
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [15, 20, 25],
        "model__max_features": ["log2", "sqrt", 10],
        "model__min_samples_leaf": [2, 3],
        "model__min_samples_split": [2, 3],
    }

    param_grid_svc = [
        {
            "model__kernel": ["rbf"],
            "model__gamma": [0.1, 0.5, 1],
            "model__C": [0.1, 1, 10],
        },
        {"model__kernel": ["linear"], "model__C": [0.1, 1, 10]},
        {
            "model__kernel": ["poly"],
            "model__degree": [2, 3],
            "model__C": [0.1, 1, 10],
        },
    ]

    best_rf = perform_grid_search(
        "Random Forest",
        pipelines["RandomForest"],
        param_grid_rf,
        X_train,
        y_train,
    )
    evaluate_model(best_rf, "Random Forest", X_val, y_val)
    save_model(best_rf.best_estimator_, models_dir / "best_rf_pipeline.pkl")

    best_svc = perform_grid_search(
        "SVC", pipelines["SVC"], param_grid_svc, X_train, y_train
    )
    evaluate_model(best_svc, "SVC", X_val, y_val)
    save_model(best_svc.best_estimator_, models_dir / "best_svc_pipeline.pkl")


if __name__ == "__main__":
    main()
