"""
Data loaders for bladder cancer recurrence datasets.
Implements single unified loader for all CSV formats (DRY principle).
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from tcris.config import settings
from tcris.utils.decorators import timer
from tcris.utils.exceptions import DataLoadingError


class DataFormat(str, Enum):
    """Enum for dataset formats."""

    WLW = "wlw"  # Wei-Lin-Weissfeld format (bladder.csv, bladder1.csv)
    AG = "ag"  # Anderson-Gill format (bladder2.csv)
    STANDARD = "standard"  # Standard format
    AUTO = "auto"  # Auto-detect format


class BladderDataLoader:
    """
    Unified data loader for all bladder cancer dataset formats.

    Handles:
    - bladder.csv: WLW format, 85 patients, up to 4 recurrences
    - bladder1.csv: Extended WLW, 118 patients, up to 9 recurrences
    - bladder2.csv: Anderson-Gill format, 85 patients

    Features:
    - Automatic format detection
    - Schema validation
    - Missing value handling
    - Type conversion
    """

    # Expected columns for each format
    SCHEMA = {
        DataFormat.WLW: {
            "bladder": ["id", "rx", "number", "size", "stop", "event", "enum"],
            "bladder1": [
                "id",
                "treatment",
                "number",
                "size",
                "recur",
                "start",
                "stop",
                "status",
                "rtumor",
                "rsize",
                "enum",
            ],
        },
        DataFormat.AG: {
            "bladder2": ["id", "rx", "number", "size", "start", "stop", "event", "enum"]
        },
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing CSV files (default: from settings)
        """
        self.data_dir = Path(data_dir) if data_dir else settings.data_path
        logger.info(f"Initialized BladderDataLoader with data_dir: {self.data_dir}")

    @timer
    def load(
        self, filename: str, format: DataFormat = DataFormat.AUTO
    ) -> Tuple[pd.DataFrame, DataFormat]:
        """
        Load a single CSV file.

        Args:
            filename: Name of CSV file (e.g., "bladder.csv")
            format: Data format (auto-detect if AUTO)

        Returns:
            Tuple of (DataFrame, detected_format)

        Raises:
            DataLoadingError: If file cannot be loaded or format is invalid
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise DataLoadingError(f"File not found: {filepath}")

        try:
            # Load CSV
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")

            # Detect format if auto
            if format == DataFormat.AUTO:
                format = self.detect_format(filename, df)
                logger.info(f"Auto-detected format: {format}")

            # Validate schema
            self._validate_schema(df, filename, format)

            # Clean data
            df = self._clean_data(df, filename, format)

            return df, format

        except Exception as e:
            raise DataLoadingError(f"Failed to load {filename}: {str(e)}")

    def load_all(self) -> Dict[str, Tuple[pd.DataFrame, DataFormat]]:
        """
        Load all dataset files.

        Returns:
            Dictionary mapping filename to (DataFrame, format)
        """
        datasets = {}
        for filename in ["bladder.csv", "bladder1.csv", "bladder2.csv"]:
            try:
                df, format = self.load(filename)
                datasets[filename] = (df, format)
            except DataLoadingError as e:
                logger.warning(f"Could not load {filename}: {e}")

        logger.info(f"Loaded {len(datasets)} datasets")
        return datasets

    def detect_format(self, filename: str, df: Optional[pd.DataFrame] = None) -> DataFormat:
        """
        Detect data format from filename and/or columns.

        Args:
            filename: Name of CSV file
            df: Optional DataFrame (for column-based detection)

        Returns:
            Detected DataFormat
        """
        # File-based detection
        if "bladder2" in filename.lower():
            return DataFormat.AG
        elif "bladder1" in filename.lower() or "bladder" in filename.lower():
            return DataFormat.WLW

        # Column-based detection
        if df is not None:
            cols = set(df.columns)
            if "start" in cols and "stop" in cols and "event" in cols:
                if "rtumor" in cols or "rsize" in cols or "status" in cols:
                    return DataFormat.WLW
                else:
                    return DataFormat.AG
            elif "stop" in cols and "event" in cols and "start" not in cols:
                return DataFormat.WLW

        return DataFormat.STANDARD

    def _validate_schema(self, df: pd.DataFrame, filename: str, format: DataFormat) -> None:
        """
        Validate DataFrame schema.

        Args:
            df: DataFrame to validate
            filename: Name of file (for error messages)
            format: Expected format

        Raises:
            DataLoadingError: If schema is invalid
        """
        # Get expected columns
        base_name = filename.replace(".csv", "")
        if format == DataFormat.WLW:
            if base_name in self.SCHEMA[DataFormat.WLW]:
                expected_cols = self.SCHEMA[DataFormat.WLW][base_name]
            else:
                # Generic WLW format check
                expected_cols = ["id", "number", "size", "stop", "event"]
        elif format == DataFormat.AG:
            if base_name in self.SCHEMA[DataFormat.AG]:
                expected_cols = self.SCHEMA[DataFormat.AG][base_name]
            else:
                expected_cols = ["id", "start", "stop", "event"]
        else:
            return  # Skip validation for STANDARD format

        # Check for required columns
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise DataLoadingError(
                f"Missing required columns in {filename}: {missing_cols}"
            )

        logger.debug(f"Schema validation passed for {filename}")

    def _clean_data(self, df: pd.DataFrame, filename: str, format: DataFormat) -> pd.DataFrame:
        """
        Clean and standardize data.

        Args:
            df: Input DataFrame
            filename: Name of file
            format: Data format

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Handle missing values (represented as "." in R datasets)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].replace(".", pd.NA)

        # Convert numeric columns
        numeric_cols = ["number", "size", "stop", "event", "enum"]
        if format == DataFormat.WLW and "bladder1" in filename:
            numeric_cols.extend(["recur", "start", "status", "rtumor", "rsize"])
        if format == DataFormat.AG:
            numeric_cols.extend(["start"])

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Standardize treatment column name
        if "rx" in df.columns and "treatment" not in df.columns:
            df["treatment"] = df["rx"]
        elif "treatment" in df.columns and "rx" not in df.columns:
            df["rx"] = df["treatment"]

        # Map treatment codes to names (if numeric)
        if "treatment" in df.columns and df["treatment"].dtype in [
            "int64",
            "float64",
        ]:
            treatment_map = {1: "placebo", 2: "thiotepa", 3: "pyridoxine"}
            df["treatment_name"] = df["treatment"].map(treatment_map)

        logger.debug(f"Data cleaning completed for {filename}")
        return df

    def get_patient_ids(self, df: pd.DataFrame) -> List[int]:
        """
        Extract unique patient IDs from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            List of unique patient IDs
        """
        if "id" not in df.columns:
            raise DataLoadingError("DataFrame does not have 'id' column")

        patient_ids = sorted(df["id"].unique().tolist())
        logger.debug(f"Found {len(patient_ids)} unique patients")
        return patient_ids

    def get_patient_data(self, df: pd.DataFrame, patient_id: int) -> pd.DataFrame:
        """
        Extract data for a specific patient.

        Args:
            df: Input DataFrame
            patient_id: Patient ID

        Returns:
            DataFrame with patient's data
        """
        if "id" not in df.columns:
            raise DataLoadingError("DataFrame does not have 'id' column")

        patient_df = df[df["id"] == patient_id].copy()

        if len(patient_df) == 0:
            raise DataLoadingError(f"Patient ID {patient_id} not found")

        return patient_df

    def get_dataset_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate summary statistics for dataset.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_patients": df["id"].nunique() if "id" in df.columns else None,
            "columns": df.columns.tolist(),
            "missing_values": df.isna().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        # Add dataset-specific statistics
        if "treatment" in df.columns or "rx" in df.columns:
            treat_col = "treatment" if "treatment" in df.columns else "rx"
            summary["treatment_distribution"] = df[treat_col].value_counts().to_dict()

        if "event" in df.columns:
            summary["event_distribution"] = df["event"].value_counts().to_dict()

        if "status" in df.columns:
            summary["status_distribution"] = df["status"].value_counts().to_dict()

        return summary
