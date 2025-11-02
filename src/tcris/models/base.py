"""Base classes for survival models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


class BaseSurvivalModel(ABC):
    """Abstract base class for survival models."""

    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        self.feature_names_ = None

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        event: np.ndarray
    ) -> "BaseSurvivalModel":
        """
        Fit the model.

        Args:
            X: Covariates
            y: Time-to-event
            event: Event indicator (1=event, 0=censored)

        Returns:
            Self
        """
        pass

    @abstractmethod
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores.

        Args:
            X: Covariates

        Returns:
            Risk scores (higher = higher risk)
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}
