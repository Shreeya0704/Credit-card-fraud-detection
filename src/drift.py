import argparse
import logging
from typing import Union

import numpy as np
import pandas as pd

from src.data import load_dataset, time_sorted_split
from src.utils import load_yaml, logger


def population_stability_index(
    ref: Union[pd.Series, np.ndarray],
    cur: Union[pd.Series, np.ndarray],
    bins: int = 20,
) -> float:
    """
    Calculate the Population Stability Index (PSI) between two distributions.

    Args:
        ref: The reference distribution (e.g., training set).
        cur: The current distribution (e.g., test set).
        bins: The number of bins to use for bucketing continuous data.

    Returns:
        The PSI value.
    """
    # Ensure inputs are numpy arrays
    ref = np.asarray(ref)
    cur = np.asarray(cur)

    # Determine bin edges from the reference distribution
    ref_min, ref_max = np.min(ref), np.max(ref)
    bin_edges = np.linspace(ref_min, ref_max, bins + 1)

    # Calculate the percentage of observations in each bin for reference and current data
    ref_counts = np.histogram(ref, bins=bin_edges)[0]
    cur_counts = np.histogram(cur, bins=bin_edges)[0]

    # Avoid division by zero
    ref_dist = ref_counts / len(ref)
    cur_dist = cur_counts / len(cur)

    # Replace zeros with a small number to avoid log(0)
    ref_dist[ref_dist == 0] = 1e-6
    cur_dist[cur_dist == 0] = 1e-6

    # Calculate PSI
    psi_values = (cur_dist - ref_dist) * np.log(cur_dist / ref_dist)
    psi = np.sum(psi_values)

    return psi


def run_drift_analysis(config_path: str):
    """CLI to run drift analysis on key features between train and test sets."""
    cfg = load_yaml(config_path)
    df = load_dataset(cfg)
    train_df, _, test_df = time_sorted_split(df, cfg)

    logger.info("Running Population Stability Index (PSI) analysis...")
    logger.info(f"Reference (train) shape: {train_df.shape}")
    logger.info(f"Current (test) shape: {test_df.shape}")

    # Features to check for drift
    features_to_check = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    for feature in features_to_check:
        psi_value = population_stability_index(train_df[feature], test_df[feature])
        
        drift_level = "INSIGNIFICANT"
        if psi_value > 0.2:
            drift_level = "SEVERE"
        elif psi_value > 0.1:
            drift_level = "MODERATE"

        logger.info(f"Feature '{feature}': PSI = {psi_value:.4f} ({drift_level} drift)")

        if drift_level in ["MODERATE", "SEVERE"]:
            logger.warning(f"Potential drift detected for feature '{feature}'. Consider retraining model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    run_drift_analysis(args.config)
