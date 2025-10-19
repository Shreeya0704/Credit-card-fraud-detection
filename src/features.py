import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TimeCyclicEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes a time column (in seconds) into cyclic sine/cosine features
    representing the time of day.
    """

    def __init__(self, time_col: str):
        self.time_col = time_col

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        # Assuming time is in seconds from a starting point, modulo by seconds in a day
        seconds_in_day = 24 * 60 * 60
        X_copy["time_of_day"] = X_copy[self.time_col] % seconds_in_day

        # Sine and Cosine transformations
        X_copy["time_sin"] = np.sin(2 * np.pi * X_copy["time_of_day"] / seconds_in_day)
        X_copy["time_cos"] = np.cos(2 * np.pi * X_copy["time_of_day"] / seconds_in_day)

        return X_copy[["time_sin", "time_cos"]]


class AmountLogTransformer(BaseEstimator, TransformerMixin):
    """Applies a log1p transformation to the 'Amount' column."""

    def __init__(self, amount_col: str = "Amount"):
        self.amount_col = amount_col

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        X_copy["amount_log"] = np.log1p(X_copy[self.amount_col])
        return X_copy[["amount_log"]]


class SequentialFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates sequential, past-only features for a given user/card ID to
    prevent data leakage in a time-sorted dataset.

    Features generated:
    - Per-card transaction count (expanding)
    - Per-card mean transaction amount (expanding)
    """

    def __init__(self, card_id_col: str, amount_col: str = "Amount"):
        self.card_id_col = card_id_col
        self.amount_col = amount_col
        self.card_counts_ = {}
        self.card_amount_sums_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit on the training data to initialize the state. This simulates
        learning from historical data before the validation/test period.
        """
        logger.info("Fitting SequentialFeatureGenerator on training data...")
        df = X.copy()
        for _, row in df.iterrows():
            card_id = row[self.card_id_col]
            amount = row[self.amount_col]

            # Update state
            self.card_counts_[card_id] = self.card_counts_.get(card_id, 0) + 1
            self.card_amount_sums_[card_id] = (
                self.card_amount_sums_.get(card_id, 0) + amount
            )
        logger.info("SequentialFeatureGenerator fit complete.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data sequentially, row by row, to generate features
        based only on past information for each card.
        """
        logger.info(f"Transforming data with SequentialFeatureGenerator...")
        df = X.copy()
        new_features = []

        for _, row in df.iterrows():
            card_id = row[self.card_id_col]
            amount = row[self.amount_col]

            # Get features from state *before* updating
            count = self.card_counts_.get(card_id, 0)
            amount_sum = self.card_amount_sums_.get(card_id, 0)
            mean_amount = amount_sum / count if count > 0 else 0

            new_features.append(
                {"card_tx_count": count, "card_mean_amount": mean_amount}
            )

            # Update state for the next transaction
            self.card_counts_[card_id] = count + 1
            self.card_amount_sums_[card_id] = amount_sum + amount

        return pd.DataFrame(new_features, index=df.index)


def build_preprocess_pipeline(
    cfg: Dict[str, Any], X: pd.DataFrame
) -> Pipeline:
    """
    Builds the feature engineering and preprocessing pipeline based on the
    config and available dataframe columns.

    Args:
        cfg: The configuration dictionary.
        X: The input feature matrix (e.g., X_train).

    Returns:
        The scikit-learn preprocessing Pipeline.
    """
    time_col = cfg["time_col"]

    # --- Define feature groups ---
    v_features = [f"V{i}" for i in range(1, 29)]
    categorical_features = [
        col for col in ["country", "mcc"] if col in X.columns
    ]
    has_card_id = "card_id" in X.columns

    # --- Build pipeline steps for ColumnTransformer ---
    base_transformers = [
        ("time_cyclic", TimeCyclicEncoder(time_col=time_col), [time_col]),
        ("amount_log", AmountLogTransformer(), ["Amount"]),
        ("v_features", "passthrough", v_features),
    ]

    # Add target encoder if categorical features are present
    if categorical_features:
        base_transformers.append(
            ("categorical", TargetEncoder(cols=categorical_features), categorical_features)
        )

    # --- Create the preprocessing pipeline ---
    preprocessor = ColumnTransformer(transformers=base_transformers, remainder="drop")

    pipeline_steps = [("base_preprocessing", preprocessor)]

    # Add sequential feature generator if card_id is present
    if has_card_id:
        pipeline_steps.append(
            ("sequential_features", SequentialFeatureGenerator(card_id_col="card_id"))
        )

    # Add optional scaling
    if cfg.get("use_standard_scaler", False):
        pipeline_steps.append(("scaler", StandardScaler()))

    pipeline = Pipeline(steps=pipeline_steps)

    logger.info("Successfully built preprocessing pipeline.")
    logger.info(f"Pipeline steps: {[step[0] for step in pipeline.steps]}")

    return pipeline
