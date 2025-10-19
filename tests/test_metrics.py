import numpy as np
import pytest

from src.metrics import recall_at_fpr, precision_at_k, expected_cost


@pytest.fixture
def sample_data():
    """Sample true labels and scores for testing metrics."""
    y_true = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.4, 0.05, 0.9, 0.15, 0.6])
    return y_true, y_score


def test_recall_at_fpr(sample_data):
    """Test the recall_at_fpr function."""
    y_true, y_score = sample_data
    # With these scores, FPR=1/6=0.166 at threshold > 0.4
    # At that point, TPR (recall) is 3/4 = 0.75
    recall = recall_at_fpr(y_true, y_score, target_fpr=0.2)
    assert np.isclose(recall, 0.75)

    # Test case where target FPR is never reached
    recall_high_fpr = recall_at_fpr(y_true, y_score, target_fpr=0.9)
    assert np.isclose(recall_high_fpr, 1.0)


def test_precision_at_k(sample_data):
    """Test the precision_at_k function."""
    y_true, y_score = sample_data
    # Top 3 scores are 0.9, 0.8, 0.7, corresponding to true labels 1, 1, 1
    precision = precision_at_k(y_true, y_score, k=3)
    assert np.isclose(precision, 1.0)

    # Top 5 scores are 0.9, 0.8, 0.7, 0.6, 0.4, corresponding to true labels 1, 1, 1, 1, 0
    precision_5 = precision_at_k(y_true, y_score, k=5)
    assert np.isclose(precision_5, 4 / 5)


def test_expected_cost(sample_data):
    """Test the expected_cost function."""
    y_true, y_score = sample_data
    cost_fp = 100
    cost_fn = 1000

    # At threshold 0.5, preds are [0,0,1,1,0,0,0,1,0,1]
    # y_true is                  [0,0,1,1,0,0,0,1,0,1]
    # No FPs or FNs
    cost = expected_cost(y_true, y_score, cost_fp, cost_fn, threshold=0.5)
    assert cost == 0

    # At threshold 0.95, preds are [0,0,0,0,0,0,0,0,0,0]
    # y_true is                   [0,0,1,1,0,0,0,1,0,1]
    # We have 4 FNs
    cost_high_thresh = expected_cost(y_true, y_score, cost_fp, cost_fn, threshold=0.95)
    assert cost_high_thresh == 4 * cost_fn

    # At threshold 0.0, preds are [1,1,1,1,1,1,1,1,1,1]
    # y_true is                   [0,0,1,1,0,0,0,1,0,1]
    # We have 6 FPs
    cost_low_thresh = expected_cost(y_true, y_score, cost_fp, cost_fn, threshold=0.0)
    assert cost_low_thresh == 6 * cost_fp
