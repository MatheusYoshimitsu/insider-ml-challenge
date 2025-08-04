from __future__ import annotations
import numpy as np
import pandas as pd
import logging


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a preconfigured logger instance.

    Args:
        name (str): Logger name (usually __name__ of the calling module).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


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
    return df.drop(columns=[col for col in drop_cols if col in df.columns])
