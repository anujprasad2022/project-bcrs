"""sklearn-compatible feature transformers pipeline."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tcris.features.extractors import TemporalFeatureExtractor, TumorProgressionExtractor


def create_feature_pipeline():
    """
    Create complete feature engineering pipeline.

    Returns:
        sklearn Pipeline
    """
    return Pipeline([
        ("temporal", TemporalFeatureExtractor()),
        ("tumor_progression", TumorProgressionExtractor()),
        ("scaler", StandardScaler())
    ])
