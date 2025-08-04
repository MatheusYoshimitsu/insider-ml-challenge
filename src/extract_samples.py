from __future__ import annotations
import logging
from pathlib import Path
from typing import NoReturn

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_api_samples(
    dataset_path: Path,
    output_path: Path,
    n_samples: int = 5,
    random_state: int = 29,
) -> None:
    """
    Extract random passenger samples from validation set and save to JSON.

    Args:
        dataset_path (Path): Path to the Titanic dataset CSV.
        output_path (Path): Path to save the JSON samples.
        n_samples (int): Number of samples to extract.
        random_state (int): Random seed for reproducibility.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    logger.info(f"Loading dataset from {dataset_path}...")
    train_df = pd.read_csv(dataset_path)

    if "Survived" not in train_df.columns:
        raise ValueError("Dataset must contain a 'Survived' column.")

    y = train_df["Survived"]
    _, X_val, _, _ = train_test_split(
        train_df, y, test_size=0.2, random_state=random_state, stratify=y
    )

    samples = X_val.sample(n_samples, random_state=random_state)
    # samples = samples.drop(columns=["Survived"], errors="ignore")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_json(output_path, orient="records", indent=4)

    logger.info(f"Saved {len(samples)} samples to {output_path}")
    logger.debug(samples.to_string())


def main() -> NoReturn:
    """
    Main entry point for extracting samples.
    """
    base_dir = Path(__file__).resolve().parents[1]
    dataset_path = base_dir / "dataset" / "train.csv"
    output_path = base_dir / "dataset" / "api_samples.json"

    extract_api_samples(dataset_path, output_path)


if __name__ == "__main__":
    main()
