from __future__ import annotations

import logging
from pathlib import Path
import joblib
import numpy as np
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
import src.train as train  # self import

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ----------------------------
# Custom Feature Engineering
# ----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform custom feature engineering on the Titanic dataset.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Transformed dataset with new features.
    """
    df = df.copy()

    # Fill missing values
    df["Embarked"] = df["Embarked"].fillna("C")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Engineered features
    df["norm_fare"] = np.log(df["Fare"] + 1)
    df["cabin_multiple"] = (
        df["Cabin"]
        .apply(lambda x: 0 if pd.isna(x) else len(x.split(" ")))
        .astype(str)
    )
    df["cabin_categories"] = df["Cabin"].apply(lambda x: str(x)[0])
    df["name_title"] = df["Name"].apply(
        lambda x: x.split(",")[1].split(".")[0].strip()
    )

    # Convert to string for categorical
    df["Pclass"] = df["Pclass"].astype(str)

    # Drop unused columns
    drop_cols = ["PassengerId", "Name", "Cabin", "Ticket", "Survived"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df


def evaluate_model(
    model: GridSearchCV, model_name: str, X_val: pd.DataFrame, y_val: pd.Series
) -> None:
    """
    Print evaluation metrics for a given model.

    Args:
        model (GridSearchCV): Trained model.
        model_name (str): Name of the model.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
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


def main() -> None:
    """Main training pipeline."""
    # ----------------------------
    # Load Data
    # ----------------------------
    dataset_path = Path("dataset/train.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    logger.info("Loading dataset...")
    train_df = pd.read_csv(dataset_path)
    y = train_df["Survived"]

    # ----------------------------
    # Preprocessing
    # ----------------------------
    categorical_cols = [
        "Pclass",
        "Sex",
        "Embarked",
        "cabin_categories",
        "cabin_multiple",
        "name_title",
    ]
    numerical_cols = ["Age", "SibSp", "Parch", "norm_fare"]

    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
            ("numerical", StandardScaler(), numerical_cols),
        ]
    )

    # Pipelines
    rf_pipeline = Pipeline(
        [
            ("feature_eng", FunctionTransformer(train.feature_engineering)),
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=29)),
        ]
    )
    svc_pipeline = Pipeline(
        [
            ("feature_eng", FunctionTransformer(train.feature_engineering)),
            ("preprocessor", preprocessor),
            ("model", SVC(probability=True, random_state=29)),
        ]
    )

    # ----------------------------
    # Split Data
    # ----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        train_df, y, test_size=0.2, random_state=29, stratify=y
    )

    # ----------------------------
    # Grid Search for Random Forest
    # ----------------------------
    param_grid_rf = {
        "model__n_estimators": [100, 200, 400],
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [15, 20, 25],
        "model__max_features": ["log2", "sqrt", 10],
        "model__min_samples_leaf": [2, 3],
        "model__min_samples_split": [2, 3],
    }

    logger.info("Starting GridSearchCV for Random Forest...")
    clf_rf = GridSearchCV(
        rf_pipeline, param_grid=param_grid_rf, cv=5, verbose=2, n_jobs=-1
    )
    best_clf_rf = clf_rf.fit(X_train, y_train)
    evaluate_model(best_clf_rf, "Random Forest", X_val, y_val)

    # ----------------------------
    # Grid Search for SVC
    # ----------------------------
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

    logger.info("Starting GridSearchCV for SVC...")
    clf_svc = GridSearchCV(
        svc_pipeline, param_grid=param_grid_svc, cv=5, verbose=2, n_jobs=-1
    )
    best_clf_svc = clf_svc.fit(X_train, y_train)
    evaluate_model(best_clf_svc, "SVC", X_val, y_val)

    # ----------------------------
    # Save Models
    # ----------------------------
    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)

    rf_path = models_dir / "best_rf_pipeline.pkl"
    svc_path = models_dir / "best_svc_pipeline.pkl"

    joblib.dump(best_clf_rf.best_estimator_, rf_path)
    joblib.dump(best_clf_svc.best_estimator_, svc_path)

    logger.info(f"Random Forest model saved to: {rf_path}")
    logger.info(f"SVC model saved to: {svc_path}")


if __name__ == "__main__":
    main()
