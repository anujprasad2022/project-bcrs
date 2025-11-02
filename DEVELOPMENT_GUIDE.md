# T-CRIS Development Guide

This guide explains how to continue developing the T-CRIS project, with detailed instructions for implementing each component.

---

## 📋 Table of Contents

1. [Development Workflow](#development-workflow)
2. [Implementing New Features](#implementing-new-features)
3. [Next Steps: Priority Implementation Guide](#next-steps-priority-implementation-guide)
4. [Code Examples & Templates](#code-examples--templates)
5. [Testing Guidelines](#testing-guidelines)
6. [Common Patterns](#common-patterns)
7. [Troubleshooting](#troubleshooting)

---

## Development Workflow

### 1. Standard Development Cycle

```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Activate Poetry shell
poetry shell

# 3. Make your changes
# - Write code
# - Write tests
# - Write docstrings

# 4. Format code
make format

# 5. Check code quality
make lint

# 6. Run tests
make test

# 7. Commit changes
git add .
git commit -m "Add feature: description"

# 8. Push and create PR
git push origin feature/your-feature-name
```

### 2. Before Committing

Always run:
```bash
make all  # Runs format + lint + test
```

---

## Implementing New Features

### Template for New Module

```python
"""
Module description.

This module provides...
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from tcris.config import settings
from tcris.utils.decorators import timer
from tcris.utils.exceptions import TCRISException


class MyNewClass:
    """
    Brief description.

    Longer description explaining what this class does,
    when to use it, and any important details.

    Attributes:
        attribute_name: Description of attribute

    Example:
        >>> obj = MyNewClass()
        >>> result = obj.process(data)
    """

    def __init__(self, param: str = "default"):
        """
        Initialize MyNewClass.

        Args:
            param: Description of parameter
        """
        self.param = param
        logger.info(f"Initialized MyNewClass with param={param}")

    @timer
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data.

        Args:
            data: Input DataFrame

        Returns:
            Processed DataFrame

        Raises:
            TCRISException: If processing fails
        """
        try:
            # Implementation here
            result = data.copy()
            # ...
            return result
        except Exception as e:
            raise TCRISException(f"Processing failed: {e}")
```

---

## Next Steps: Priority Implementation Guide

### PRIORITY 1: Data Validation (1-2 days)

**File**: `src/tcris/data/validators.py`

**What to Implement**:

```python
"""Data validation using Great Expectations."""

import great_expectations as ge
from typing import Dict, List
import pandas as pd


class DataValidator:
    """Validate bladder cancer data quality."""

    def __init__(self):
        self.expectations = self._create_expectations()

    def _create_expectations(self) -> List:
        """Define data quality expectations."""
        return [
            # Completeness
            ("expect_column_values_to_not_be_null", {"column": "id"}),
            ("expect_column_values_to_not_be_null", {"column": "treatment"}),

            # Validity
            ("expect_column_values_to_be_in_set", {
                "column": "treatment",
                "value_set": ["placebo", "pyridoxine", "thiotepa"]
            }),
            ("expect_column_values_to_be_between", {
                "column": "number",
                "min_value": 1,
                "max_value": 20
            }),

            # Temporal consistency
            # Add more expectations...
        ]

    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Validate DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Validation report dictionary
        """
        ge_df = ge.from_pandas(df)

        results = []
        for expectation_name, kwargs in self.expectations:
            method = getattr(ge_df, expectation_name)
            result = method(**kwargs)
            results.append(result)

        # Compile report
        report = {
            "is_valid": all(r.success for r in results),
            "n_checks": len(results),
            "n_passed": sum(r.success for r in results),
            "n_failed": sum(not r.success for r in results),
            "failures": [r for r in results if not r.success]
        }

        return report
```

**Test File**: `tests/unit/test_validators.py`

```python
import pytest
import pandas as pd
from tcris.data.validators import DataValidator


class TestDataValidator:
    def test_valid_data_passes(self):
        """Test that valid data passes all checks."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "treatment": ["placebo", "thiotepa", "pyridoxine"],
            "number": [2, 3, 1],
            "size": [1.5, 2.0, 1.0]
        })

        validator = DataValidator()
        report = validator.validate(df)

        assert report["is_valid"] is True
        assert report["n_failed"] == 0

    def test_invalid_treatment_fails(self):
        """Test that invalid treatment values are caught."""
        df = pd.DataFrame({
            "id": [1],
            "treatment": ["invalid"],
            "number": [2],
            "size": [1.5]
        })

        validator = DataValidator()
        report = validator.validate(df)

        assert report["is_valid"] is False
        assert report["n_failed"] > 0
```

**How to Run**:
```bash
poetry run pytest tests/unit/test_validators.py -v
```

---

### PRIORITY 2: Feature Engineering (2-3 days)

**File**: `src/tcris/features/extractors.py`

**What to Implement**:

```python
"""Feature extraction for bladder cancer data."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
        - acceleration (change in inter-recurrence times)
        - time_to_first_recurrence
        """
        X = X.copy()

        # Group by patient
        for patient_id in X["patient_id"].unique():
            mask = X["patient_id"] == patient_id
            patient_df = X[mask].sort_values("event_number")

            # Time since last recurrence
            X.loc[mask, "time_since_last"] = patient_df["start_time"].diff()

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

        return X
```

**File**: `src/tcris/features/transformers.py`

```python
"""sklearn-compatible feature transformers pipeline."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tcris.features.extractors import (
    TemporalFeatureExtractor,
    TumorProgressionExtractor
)


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
```

---

### PRIORITY 3: Cox PH Model (1-2 days)

**File**: `src/tcris/models/base.py`

```python
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
    def predict_survival(
        self,
        X: pd.DataFrame,
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict survival probabilities.

        Args:
            X: Covariates
            times: Time points for prediction

        Returns:
            Survival probabilities array
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
```

**File**: `src/tcris/models/statistical/cox.py`

```python
"""Cox Proportional Hazards model."""

from typing import Optional
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from loguru import logger

from tcris.models.base import BaseSurvivalModel
from tcris.utils.decorators import timer


class CoxPHModel(BaseSurvivalModel):
    """
    Cox Proportional Hazards model.

    Wraps lifelines.CoxPHFitter with consistent interface.

    Attributes:
        penalizer: L2 regularization parameter
        model: Fitted CoxPHFitter instance
    """

    def __init__(self, penalizer: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.penalizer = penalizer
        self.model = CoxPHFitter(penalizer=penalizer)

    @timer
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        event: np.ndarray
    ) -> "CoxPHModel":
        """Fit Cox PH model."""
        # Prepare data for lifelines
        df = X.copy()
        df["duration"] = y
        df["event"] = event

        # Fit model
        self.model.fit(
            df,
            duration_col="duration",
            event_col="event",
            show_progress=False
        )

        self.is_fitted = True
        logger.info("Cox PH model fitted successfully")
        logger.info(f"C-index: {self.model.concordance_index_:.3f}")

        return self

    def predict_survival(
        self,
        X: pd.DataFrame,
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict survival probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if times is None:
            times = np.linspace(0, X["duration"].max(), 100)

        survival_probs = self.model.predict_survival_function(
            X,
            times=times
        )

        return survival_probs.values

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk scores (partial hazard)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_partial_hazard(X).values

    def get_summary(self) -> pd.DataFrame:
        """Get model summary with coefficients and p-values."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted")

        return self.model.summary

    def plot_coefficients(self):
        """Plot hazard ratios with confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted")

        return self.model.plot()
```

**Usage Example**:

```python
from tcris.models.statistical.cox import CoxPHModel
from tcris.data.loaders import BladderDataLoader
from tcris.data.fusion import DataFusionEngine

# Load and prepare data
loader = BladderDataLoader()
datasets = loader.load_all()
fusion = DataFusionEngine()
unified_df = fusion.fuse(datasets)

# Prepare for Cox model
X = unified_df[["treatment", "baseline_tumors", "baseline_size"]]
X = pd.get_dummies(X, columns=["treatment"], drop_first=True)

y = unified_df["stop_time"]
event = (unified_df["event_type"] == 1).astype(int)

# Fit model
model = CoxPHModel(penalizer=0.01)
model.fit(X, y, event)

# Get summary
print(model.get_summary())

# Predict risk for new patient
new_patient = pd.DataFrame({
    "baseline_tumors": [3],
    "baseline_size": [2.0],
    "treatment_thiotepa": [1],
    "treatment_pyridoxine": [0]
})

risk_score = model.predict_risk(new_patient)
print(f"Risk score: {risk_score[0]:.3f}")
```

---

### PRIORITY 4: Visualization (1-2 days)

**File**: `src/tcris/visualization/survival_plots.py`

```python
"""Survival analysis visualizations."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter
from loguru import logger


def plot_kaplan_meier(
    data: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event",
    group_col: Optional[str] = None,
    title: str = "Kaplan-Meier Survival Curves"
) -> go.Figure:
    """
    Plot Kaplan-Meier survival curves.

    Args:
        data: DataFrame with survival data
        duration_col: Column name for time-to-event
        event_col: Column name for event indicator
        group_col: Optional column for grouping (e.g., treatment)
        title: Plot title

    Returns:
        Plotly Figure
    """
    kmf = KaplanMeierFitter()
    fig = go.Figure()

    if group_col is None:
        # Single curve
        kmf.fit(data[duration_col], data[event_col], label="All")

        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_["All"],
            mode="lines",
            name="All patients",
            line=dict(width=2)
        ))

        # Confidence intervals
        ci = kmf.confidence_interval_survival_function_
        fig.add_trace(go.Scatter(
            x=ci.index,
            y=ci.iloc[:, 0],
            fill=None,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=ci.index,
            y=ci.iloc[:, 1],
            fill="tonexty",
            mode="lines",
            line=dict(width=0),
            name="95% CI",
            fillcolor="rgba(0,100,200,0.2)"
        ))

    else:
        # Multiple curves by group
        groups = data[group_col].unique()
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, group in enumerate(groups):
            mask = data[group_col] == group
            kmf.fit(
                data.loc[mask, duration_col],
                data.loc[mask, event_col],
                label=str(group)
            )

            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_[str(group)],
                mode="lines",
                name=str(group),
                line=dict(width=2, color=colors[i % len(colors)])
            ))

    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (months)",
        yaxis_title="Survival Probability",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )

    return fig


def plot_cumulative_incidence(
    data: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "event_type",
    title: str = "Cumulative Incidence Functions"
) -> go.Figure:
    """
    Plot cumulative incidence for competing risks.

    Args:
        data: DataFrame with competing risks data
        duration_col: Column name for time
        event_col: Column name for event type
        title: Plot title

    Returns:
        Plotly Figure
    """
    event_types = data[event_col].unique()
    event_types = [e for e in event_types if e != 0]  # Exclude censored

    fig = go.Figure()

    for event_type in sorted(event_types):
        # Calculate cumulative incidence for this event type
        # (simplified - use proper competing risks calculation in production)
        times = np.sort(data[data[event_col] == event_type][duration_col].unique())
        cum_incidence = []

        for t in times:
            # Proportion who had this event by time t
            at_risk = (data[duration_col] >= t).sum()
            had_event = ((data[duration_col] <= t) & (data[event_col] == event_type)).sum()
            cum_inc = had_event / len(data) if len(data) > 0 else 0
            cum_incidence.append(cum_inc)

        event_labels = {
            1: "Recurrence",
            2: "Death (Bladder Cancer)",
            3: "Death (Other)"
        }

        fig.add_trace(go.Scatter(
            x=times,
            y=cum_incidence,
            mode="lines",
            name=event_labels.get(event_type, f"Event {event_type}"),
            line=dict(width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (months)",
        yaxis_title="Cumulative Incidence",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )

    return fig
```

---

### PRIORITY 5: Basic Dashboard (2-3 days)

**File**: `dashboard/app.py`

```python
"""Main Streamlit dashboard for T-CRIS."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcris.data.loaders import BladderDataLoader
from tcris.data.fusion import DataFusionEngine


# Page config
st.set_page_config(
    page_title="T-CRIS Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏥 T-CRIS: Temporal Cancer Recurrence Intelligence System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["Overview", "Survival Analysis", "Predictions", "Counterfactual", "Interpretability"]
    )

    st.markdown("---")
    st.header("Settings")
    show_raw_data = st.checkbox("Show Raw Data", value=False)

# Load data (cached)
@st.cache_data
def load_data():
    """Load and fuse data."""
    loader = BladderDataLoader()
    datasets = loader.load_all()
    fusion = DataFusionEngine()
    unified_df = fusion.fuse(datasets)
    return unified_df, fusion

unified_df, fusion = load_data()

# Page routing
if page == "Overview":
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Patients", unified_df["patient_id"].nunique())

    with col2:
        st.metric("Total Events", len(unified_df))

    with col3:
        recurrence_counts = fusion.get_recurrence_counts(unified_df)
        st.metric("Mean Recurrences", f"{recurrence_counts.mean():.1f}")

    with col4:
        max_follow_up = unified_df.groupby("patient_id")["stop_time"].max().max()
        st.metric("Max Follow-up (months)", f"{max_follow_up:.0f}")

    st.markdown("---")

    # Summary statistics
    st.subheader("Summary Statistics")
    summary = fusion.summarize_unified_data(unified_df)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Treatment Distribution**")
        st.json(summary["treatment_distribution"])

    with col2:
        st.write("**Event Type Distribution**")
        st.json(summary["event_type_distribution"])

    # Show raw data
    if show_raw_data:
        st.subheader("Raw Data")
        st.dataframe(unified_df.head(100))

elif page == "Survival Analysis":
    st.header("📈 Survival Analysis")

    st.info("Survival analysis visualizations will be implemented here.")
    st.write("Features:")
    st.write("- Kaplan-Meier curves by treatment")
    st.write("- Log-rank test results")
    st.write("- Cumulative incidence functions")
    st.write("- Hazard ratio plots")

elif page == "Predictions":
    st.header("🎯 Individual Predictions")

    st.info("Prediction interface will be implemented here.")

    # Example input form
    st.subheader("Patient Characteristics")

    col1, col2 = st.columns(2)

    with col1:
        number_tumors = st.slider("Initial Tumor Count", 1, 10, 2)
        largest_size = st.number_input("Largest Tumor Size (cm)", 0.1, 10.0, 1.5)

    with col2:
        treatment = st.selectbox("Treatment", ["placebo", "pyridoxine", "thiotepa"])

    if st.button("Predict"):
        st.write("Prediction results will appear here.")

elif page == "Counterfactual":
    st.header("🔀 Counterfactual Analysis")
    st.info("Treatment comparison will be implemented here.")

elif page == "Interpretability":
    st.header("🔍 Model Interpretability")
    st.info("SHAP and attention visualizations will be implemented here.")

# Footer
st.markdown("---")
st.markdown("**T-CRIS** - Temporal Cancer Recurrence Intelligence System | Built with Streamlit")
```

**Run Dashboard**:
```bash
make dashboard
# or
poetry run streamlit run dashboard/app.py
```

---

## Testing Guidelines

### Writing Good Tests

1. **Test one thing at a time**
2. **Use descriptive names**: `test_cox_model_predicts_higher_risk_for_larger_tumors`
3. **Follow AAA pattern**: Arrange, Act, Assert
4. **Use fixtures for common setup**

### Example Test Structure

```python
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    """Create sample survival data."""
    return pd.DataFrame({
        "patient_id": [1, 1, 2, 2, 3],
        "treatment": ["placebo", "placebo", "thiotepa", "thiotepa", "placebo"],
        "baseline_tumors": [2, 2, 3, 3, 1],
        "baseline_size": [1.5, 1.5, 2.0, 2.0, 1.0],
        "duration": [10, 20, 5, 15, 30],
        "event": [1, 0, 1, 0, 0]
    })


class TestCoxPHModel:
    """Tests for Cox PH model."""

    def test_model_fits_successfully(self, sample_data):
        """Test that model fits without errors."""
        # Arrange
        from tcris.models.statistical.cox import CoxPHModel

        X = sample_data[["treatment", "baseline_tumors", "baseline_size"]]
        X = pd.get_dummies(X, columns=["treatment"])
        y = sample_data["duration"]
        event = sample_data["event"]

        # Act
        model = CoxPHModel()
        model.fit(X, y, event)

        # Assert
        assert model.is_fitted is True
        assert model.model is not None

    def test_model_predicts_risk_scores(self, sample_data):
        """Test that model produces risk scores."""
        # Similar structure...
        pass
```

---

## Common Patterns

### 1. Loading Data

```python
from tcris.data.loaders import BladderDataLoader
from tcris.data.fusion import DataFusionEngine

loader = BladderDataLoader()
datasets = loader.load_all()

fusion = DataFusionEngine()
unified_df = fusion.fuse(datasets)
```

### 2. Feature Engineering

```python
from tcris.features.transformers import create_feature_pipeline

pipeline = create_feature_pipeline()
X_transformed = pipeline.fit_transform(X)
```

### 3. Model Training

```python
from tcris.models.statistical.cox import CoxPHModel

model = CoxPHModel(penalizer=0.01)
model.fit(X_train, y_train, event_train)

# Evaluate
from tcris.evaluation.metrics import concordance_index

c_index = concordance_index(y_test, model.predict_risk(X_test), event_test)
print(f"C-index: {c_index:.3f}")
```

### 4. Visualization

```python
from tcris.visualization.survival_plots import plot_kaplan_meier

fig = plot_kaplan_meier(
    data=df,
    duration_col="duration",
    event_col="event",
    group_col="treatment"
)

fig.show()
# or in Streamlit:
st.plotly_chart(fig)
```

---

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'tcris'`

**Solution**:
```bash
# Make sure you're in Poetry shell
poetry shell

# Or use poetry run
poetry run python your_script.py
```

### Type Errors

**Problem**: mypy complains about types

**Solution**:
```bash
# Check specific file
poetry run mypy src/tcris/your_file.py

# Common fix: add type hints
def my_function(x: int) -> str:
    return str(x)
```

### Test Failures

**Problem**: Tests fail after changes

**Solution**:
```bash
# Run specific test with verbose output
poetry run pytest tests/unit/test_your_file.py::test_your_function -v

# Run with debugger
poetry run pytest tests/unit/test_your_file.py::test_your_function --pdb
```

---

## Summary

### Recommended Development Order

1. ✅ **Week 1**: Data validation + Feature engineering
2. ✅ **Week 2**: Statistical models (Cox, AG) + Evaluation metrics
3. ✅ **Week 3**: ML models (RSF, GBM) + Basic dashboard
4. ✅ **Week 4**: Deep learning models (LSTM, Transformer)
5. ✅ **Week 5**: Prediction engine + Interpretability
6. ✅ **Week 6**: Counterfactual analysis + API
7. ✅ **Week 7**: Dashboard polish + Reports
8. ✅ **Week 8**: Testing + Documentation + Presentation prep

### Key Principles

- **Start simple**: Get basic version working first
- **Test early**: Write tests as you go
- **Document everything**: Docstrings for all public functions
- **Follow patterns**: Use existing code as templates
- **Ask for help**: Check documentation, GitHub issues

---

**Happy Coding! 🚀**

For questions or issues, refer to:
- [PROJECT_README.md](PROJECT_README.md)
- [INSTALLATION.md](INSTALLATION.md)
- [PROJECT_STATUS.md](PROJECT_STATUS.md)
