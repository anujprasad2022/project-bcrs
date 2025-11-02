"""Custom exceptions for T-CRIS."""


class TCRISException(Exception):
    """Base exception for all T-CRIS errors."""

    pass


class DataValidationError(TCRISException):
    """Raised when data validation fails."""

    pass


class ModelNotFoundError(TCRISException):
    """Raised when a requested model is not found."""

    pass


class PredictionError(TCRISException):
    """Raised when prediction fails."""

    pass


class DataLoadingError(TCRISException):
    """Raised when data loading fails."""

    pass


class FeatureEngineeringError(TCRISException):
    """Raised when feature engineering fails."""

    pass
