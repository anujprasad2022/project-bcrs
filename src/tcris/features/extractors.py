"""Feature extraction for bladder cancer recurrence data."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from recurrence data."""

    def fit(self, X: pd.DataFrame, y=None):
        """Fit extractor (no-op for this transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features.

        Features:
        - time_since_last_recurrence
        - recurrence_rate
        - time_to_first_recurrence
        """
        X = X.copy()

        # Group by patient
        for patient_id in X["patient_id"].unique():
            mask = X["patient_id"] == patient_id
            patient_df = X[mask].sort_values("event_number")

            # Time since last recurrence
            X.loc[mask, "time_since_last"] = patient_df["start_time"].diff().fillna(0)

            # Recurrence rate (recurrences / follow-up time)
            total_time = patient_df["stop_time"].max()
            n_recurrences = (patient_df["event_type"] == 1).sum()
            X.loc[mask, "recurrence_rate"] = n_recurrences / (total_time + 1e-6)

            # Time to first recurrence
            first_recurrence_time = patient_df[
                patient_df["event_type"] == 1
            ]["stop_time"].min()
            X.loc[mask, "time_to_first_recurrence"] = (
                first_recurrence_time if pd.notna(first_recurrence_time) else total_time
            )

        logger.debug("Temporal features extracted")
        return X


class TumorProgressionExtractor(BaseEstimator, TransformerMixin):
    """Extract tumor progression features."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract tumor progression features.

        Features:
        - tumor_count_change (current - baseline)
        - tumor_size_change (current - baseline)
        - tumor_burden_index (count × size)
        - progression_velocity (change / time)
        """
        X = X.copy()

        # Tumor changes
        X["tumor_count_change"] = X["current_tumors"] - X["baseline_tumors"]
        X["tumor_size_change"] = X["current_size"] - X["baseline_size"]

        # Tumor burden
        X["tumor_burden_index"] = X["current_tumors"] * X["current_size"]
        X["baseline_burden"] = X["baseline_tumors"] * X["baseline_size"]

        # Progression velocity
        time_elapsed = X["stop_time"] - X["start_time"] + 1e-6
        X["count_velocity"] = X["tumor_count_change"] / time_elapsed
        X["size_velocity"] = X["tumor_size_change"] / time_elapsed

        logger.debug("Tumor progression features extracted")
        return X


def create_features(unified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features for modeling.

    Args:
        unified_df: Unified DataFrame from fusion

    Returns:
        DataFrame with all features
    """
    # Extract features
    temporal_extractor = TemporalFeatureExtractor()
    tumor_extractor = TumorProgressionExtractor()

    df_features = temporal_extractor.transform(unified_df)
    df_features = tumor_extractor.transform(df_features)

    # One-hot encode treatment
    df_features = pd.get_dummies(df_features, columns=["treatment"], prefix="treat", drop_first=False)

    # Fill any remaining NaN values
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    df_features[numeric_cols] = df_features[numeric_cols].fillna(0)

    logger.info(f"Created {len(df_features.columns)} features")

    return df_features
