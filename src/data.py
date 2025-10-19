import logging
from typing import Any, Dict, Tuple

import pandas as pd

from src.utils import timer

logger = logging.getLogger(__name__)


@timer
def load_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load the dataset from the specified CSV file and validate its schema.

    Args:
        cfg: The configuration dictionary, expecting 'dataset_path'.

    Returns:
        The loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the dataset file is not found.
        ValueError: If the dataset is missing expected columns.
    """
    dataset_path = cfg["dataset_path"]
    logger.info(f"Loading dataset from: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        logger.error(f"Dataset file not found at: {dataset_path}")
        raise

    # Validate the base Kaggle schema
    expected_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Dataset is missing expected Kaggle columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info("Base Kaggle schema validated successfully.")

    # Check for the presence of optional richer feature fields
    rich_cols = ["card_id", "merchant_id", "country", "mcc"]
    found_rich_cols = [col for col in rich_cols if col in df.columns]
    if found_rich_cols:
        logger.info(f"Found richer feature columns: {found_rich_cols}")
    else:
        logger.info("No richer feature columns found. Proceeding with base schema.")

    return df


@timer
def time_sorted_split(
    df: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sorts the dataframe by time and splits it chronologically into train,
    validation, and test sets.

    Args:
        df: The input pandas DataFrame, must contain the time column.
        cfg: The configuration dictionary, expecting 'time_col' and
             'train_valid_test_ratios'.

    Returns:
        A tuple containing the train, validation, and test DataFrames.

    Raises:
        ValueError: If time leakage is detected between the splits.
    """
    time_col = cfg["time_col"]
    ratios = cfg["train_valid_test_ratios"]
    logger.info(f"Performing time-sorted split with ratios: {ratios}")

    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)

    train_ratio, valid_ratio, _ = ratios
    n_samples = len(df_sorted)

    train_end_idx = int(n_samples * train_ratio)
    valid_end_idx = train_end_idx + int(n_samples * valid_ratio)

    train_df = df_sorted.iloc[:train_end_idx]
    valid_df = df_sorted.iloc[train_end_idx:valid_end_idx]
    test_df = df_sorted.iloc[valid_end_idx:]

    # Assert no future leakage into the past
    try:
        assert train_df[time_col].max() < valid_df[time_col].min()
        assert valid_df[time_col].max() <= test_df[time_col].min()
        logger.info("Chronological split validated. No time leakage detected.")
    except AssertionError:
        error_msg = "Time leakage detected in chronological split! Overlap in time between sets."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Valid shape: {valid_df.shape}")
    logger.info(f"Test shape:  {test_df.shape}")

    return train_df, valid_df, test_df
