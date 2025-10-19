import logging
from typing import Any, Dict, Tuple

import numpy as np

from src.metrics import expected_cost
from src.utils import load_yaml, save_yaml, timer

logger = logging.getLogger(__name__)


@timer
def find_best_threshold_by_cost(
    y_true: np.ndarray, y_score: np.ndarray, cost_fp: float, cost_fn: float
) -> Tuple[float, float]:
    """
    Find the best probability threshold by minimizing the expected cost.

    Args:
        y_true: True labels.
        y_score: Predicted scores.
        cost_fp: The cost of a single false positive.
        cost_fn: The cost of a single false negative.

    Returns:
        A tuple containing the best threshold and the minimum cost.
    """
    thresholds = np.linspace(0.0, 1.0, 500)
    costs = [
        expected_cost(y_true, y_score, cost_fp, cost_fn, t) for t in thresholds
    ]

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    min_cost = costs[best_idx]

    logger.info(f"Best threshold by cost: {best_threshold:.4f} with min cost: {min_cost:,.2f}")
    return best_threshold, min_cost


@timer
def threshold_at_top_k(y_score: np.ndarray, k: int) -> float:
    """
    Find the probability threshold that selects the top K riskiest transactions.

    Args:
        y_score: Predicted scores.
        k: The number of top transactions to select.

    Returns:
        The probability threshold.
    """
    if k >= len(y_score):
        logger.warning(f"K ({k}) is >= number of samples ({len(y_score)}). Returning threshold of 0.0")
        return 0.0

    threshold = np.sort(y_score)[-k]
    logger.info(f"Threshold for top {k} transactions: {threshold:.4f}")
    return threshold


@timer
def persist_thresholds(cfg_path: str, thresholds: Dict[str, Any]):
    """
    Saves the calculated thresholds to the specified YAML config file.

    Args:
        cfg_path: Path to the thresholds YAML file.
        thresholds: A dictionary containing the thresholds to save.
    """
    try:
        # Convert numpy types to native python types
        for k, v in thresholds.items():
            if isinstance(v, np.generic):
                thresholds[k] = v.item()

        # Load existing thresholds to preserve any other values
        try:
            existing_thresholds = load_yaml(cfg_path)
            if existing_thresholds is None:
                existing_thresholds = {}
        except FileNotFoundError:
            existing_thresholds = {}
            
        existing_thresholds.update(thresholds)
        save_yaml(cfg_path, existing_thresholds)
        logger.info(f"Successfully persisted thresholds to {cfg_path}")
    except Exception as e:
        logger.error(f"Failed to persist thresholds: {e}")
        raise
