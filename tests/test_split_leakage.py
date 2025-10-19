import pandas as pd
import pytest

from src.data import time_sorted_split


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        "Time": range(100),
        "Feature": range(100),
        "Class": [0] * 90 + [1] * 10, # 10% fraud rate
    }
    return pd.DataFrame(data)


def test_time_sorted_split_no_leakage(sample_dataframe):
    """Test that the chronological split does not leak future data into the past."""
    cfg = {
        "time_col": "Time",
        "train_valid_test_ratios": [0.7, 0.15, 0.15],
    }
    train_df, valid_df, test_df = time_sorted_split(sample_dataframe, cfg)

    assert not train_df.empty
    assert not valid_df.empty
    assert not test_df.empty

    # The core assertion: max time in train is less than min time in valid, etc.
    assert train_df["Time"].max() < valid_df["Time"].min()
    assert valid_df["Time"].max() < test_df["Time"].min()


def test_time_sorted_split_ratios(sample_dataframe):
    """Test that the split ratios are approximately correct."""
    cfg = {
        "time_col": "Time",
        "train_valid_test_ratios": [0.7, 0.15, 0.15],
    }
    train_df, valid_df, test_df = time_sorted_split(sample_dataframe, cfg)

    total_len = len(sample_dataframe)
    assert len(train_df) == int(total_len * 0.7)
    assert len(valid_df) == int(total_len * 0.15)
    # The test set takes the remainder
    assert len(test_df) == total_len - int(total_len * 0.7) - int(total_len * 0.15)
