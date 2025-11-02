"""
Data fusion engine to unify multiple dataset formats.
Converts WLW, Anderson-Gill, and standard formats into unified temporal representation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from tcris.data.loaders import DataFormat
from tcris.utils.decorators import timer
from tcris.utils.exceptions import DataLoadingError


class DataFusionEngine:
    """
    Unifies multiple bladder cancer dataset formats into coherent temporal representation.

    Unified schema:
    - patient_id: unique identifier
    - start_time: interval start
    - stop_time: interval end
    - event_type: {0: censored, 1: recurrence, 2: death_bladder, 3: death_other}
    - event_number: sequential event count (1, 2, 3, ...)
    - treatment: treatment arm (placebo, pyridoxine, thiotepa)
    - baseline_tumors: initial tumor count
    - baseline_size: initial largest tumor size
    - current_tumors: tumor count at this interval
    - current_size: largest tumor size at this interval
    - format_source: which dataset it came from
    """

    UNIFIED_COLUMNS = [
        "patient_id",
        "start_time",
        "stop_time",
        "event_type",
        "event_number",
        "treatment",
        "baseline_tumors",
        "baseline_size",
        "current_tumors",
        "current_size",
        "format_source",
    ]

    @timer
    def fuse(
        self, datasets: Dict[str, Tuple[pd.DataFrame, DataFormat]]
    ) -> pd.DataFrame:
        """
        Fuse multiple datasets into unified format.

        Args:
            datasets: Dictionary mapping filename to (DataFrame, DataFormat)

        Returns:
            Unified DataFrame

        Raises:
            DataLoadingError: If fusion fails
        """
        unified_dfs = []

        for filename, (df, format) in datasets.items():
            logger.info(f"Converting {filename} ({format}) to unified format")

            if format == DataFormat.WLW:
                if "bladder1" in filename:
                    unified_df = self._convert_bladder1(df, filename)
                else:
                    unified_df = self._convert_bladder(df, filename)
            elif format == DataFormat.AG:
                unified_df = self._convert_bladder2(df, filename)
            else:
                logger.warning(f"Unknown format for {filename}, skipping")
                continue

            unified_dfs.append(unified_df)

        if not unified_dfs:
            raise DataLoadingError("No datasets could be converted")

        # Concatenate all datasets
        result = pd.concat(unified_dfs, ignore_index=True)

        logger.info(
            f"Fused {len(datasets)} datasets into {len(result)} rows, "
            f"{result['patient_id'].nunique()} unique patients"
        )

        return result

    def _convert_bladder(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Convert bladder.csv (WLW format) to unified format.

        Original columns: id, rx, number, size, stop, event, enum
        """
        records = []

        for patient_id in df["id"].unique():
            patient_df = df[df["id"] == patient_id].sort_values("enum")

            # Get baseline characteristics
            baseline_row = patient_df.iloc[0]
            treatment = self._map_treatment(baseline_row.get("rx", baseline_row.get("treatment")))
            baseline_tumors = baseline_row["number"]
            baseline_size = baseline_row["size"]

            # Process each event (potential recurrence)
            prev_stop = 0
            for idx, row in patient_df.iterrows():
                event_num = int(row["enum"])
                stop_time = row["stop"]
                event = int(row["event"])

                record = {
                    "patient_id": int(patient_id),
                    "start_time": prev_stop,
                    "stop_time": stop_time,
                    "event_type": event,  # 0=censored, 1=recurrence
                    "event_number": event_num,
                    "treatment": treatment,
                    "baseline_tumors": baseline_tumors,
                    "baseline_size": baseline_size,
                    "current_tumors": baseline_tumors,  # Not available in this dataset
                    "current_size": baseline_size,  # Not available
                    "format_source": source,
                }
                records.append(record)
                prev_stop = stop_time

        return pd.DataFrame(records, columns=self.UNIFIED_COLUMNS)

    def _convert_bladder1(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Convert bladder1.csv (extended WLW format) to unified format.

        Original columns: id, treatment, number, size, recur, start, stop,
                         status, rtumor, rsize, enum
        """
        records = []

        for patient_id in df["id"].unique():
            patient_df = df[df["id"] == patient_id].sort_values("enum")

            # Get baseline characteristics
            baseline_row = patient_df.iloc[0]
            treatment = self._map_treatment(baseline_row["treatment"])
            baseline_tumors = baseline_row["number"]
            baseline_size = baseline_row["size"]

            # Process each event
            for idx, row in patient_df.iterrows():
                event_num = int(row["enum"]) if pd.notna(row["enum"]) else 0
                start_time = row.get("start", 0)
                stop_time = row["stop"]
                status = int(row["status"]) if pd.notna(row["status"]) else 0

                # Current tumor characteristics (at recurrence)
                current_tumors = row.get("rtumor", baseline_tumors)
                current_size = row.get("rsize", baseline_size)

                # Handle missing values
                if pd.isna(current_tumors):
                    current_tumors = baseline_tumors
                if pd.isna(current_size):
                    current_size = baseline_size

                record = {
                    "patient_id": int(patient_id),
                    "start_time": start_time if pd.notna(start_time) else 0,
                    "stop_time": stop_time,
                    "event_type": status,  # 0=censored, 1=recurrence, 2=death_bladder, 3=death_other
                    "event_number": event_num,
                    "treatment": treatment,
                    "baseline_tumors": baseline_tumors,
                    "baseline_size": baseline_size,
                    "current_tumors": float(current_tumors),
                    "current_size": float(current_size),
                    "format_source": source,
                }
                records.append(record)

        return pd.DataFrame(records, columns=self.UNIFIED_COLUMNS)

    def _convert_bladder2(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Convert bladder2.csv (Anderson-Gill format) to unified format.

        Original columns: id, rx, number, size, start, stop, event, enum
        """
        records = []

        for patient_id in df["id"].unique():
            patient_df = df[df["id"] == patient_id].sort_values("enum")

            # Get baseline characteristics
            baseline_row = patient_df.iloc[0]
            treatment = self._map_treatment(baseline_row.get("rx", baseline_row.get("treatment")))
            baseline_tumors = baseline_row["number"]
            baseline_size = baseline_row["size"]

            # Process each interval
            for idx, row in patient_df.iterrows():
                event_num = int(row["enum"])
                start_time = row["start"]
                stop_time = row["stop"]
                event = int(row["event"])

                record = {
                    "patient_id": int(patient_id),
                    "start_time": start_time,
                    "stop_time": stop_time,
                    "event_type": event,
                    "event_number": event_num,
                    "treatment": treatment,
                    "baseline_tumors": baseline_tumors,
                    "baseline_size": baseline_size,
                    "current_tumors": baseline_tumors,  # Not available
                    "current_size": baseline_size,  # Not available
                    "format_source": source,
                }
                records.append(record)

        return pd.DataFrame(records, columns=self.UNIFIED_COLUMNS)

    def _map_treatment(self, treatment_value: any) -> str:
        """
        Map treatment code to name.

        Args:
            treatment_value: Treatment code or name

        Returns:
            Standardized treatment name
        """
        if pd.isna(treatment_value):
            return "unknown"

        # If already a string, return as-is
        if isinstance(treatment_value, str):
            return treatment_value.lower()

        # Map numeric codes
        treatment_map = {
            1: "placebo",
            2: "thiotepa",
            3: "pyridoxine",
        }

        return treatment_map.get(int(treatment_value), "unknown")

    def get_patient_trajectory(
        self, unified_df: pd.DataFrame, patient_id: int
    ) -> pd.DataFrame:
        """
        Extract full temporal trajectory for a patient.

        Args:
            unified_df: Unified DataFrame
            patient_id: Patient ID

        Returns:
            DataFrame with patient's complete trajectory
        """
        trajectory = unified_df[unified_df["patient_id"] == patient_id].sort_values(
            "event_number"
        )

        if len(trajectory) == 0:
            raise DataLoadingError(f"Patient ID {patient_id} not found in unified data")

        return trajectory

    def get_recurrence_counts(self, unified_df: pd.DataFrame) -> pd.Series:
        """
        Count recurrences per patient.

        Args:
            unified_df: Unified DataFrame

        Returns:
            Series with patient_id as index and recurrence count as values
        """
        recurrence_counts = (
            unified_df[unified_df["event_type"] == 1]
            .groupby("patient_id")
            .size()
        )

        # Include patients with zero recurrences
        all_patients = unified_df["patient_id"].unique()
        recurrence_counts = recurrence_counts.reindex(all_patients, fill_value=0)

        return recurrence_counts

    def summarize_unified_data(self, unified_df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate summary statistics for unified data.

        Args:
            unified_df: Unified DataFrame

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "n_rows": len(unified_df),
            "n_patients": unified_df["patient_id"].nunique(),
            "treatment_distribution": unified_df.groupby("patient_id")["treatment"]
            .first()
            .value_counts()
            .to_dict(),
            "event_type_distribution": unified_df["event_type"].value_counts().to_dict(),
            "recurrence_counts": self.get_recurrence_counts(unified_df).describe().to_dict(),
            "follow_up_time": {
                "mean": unified_df.groupby("patient_id")["stop_time"].max().mean(),
                "median": unified_df.groupby("patient_id")["stop_time"].max().median(),
                "max": unified_df.groupby("patient_id")["stop_time"].max().max(),
            },
            "baseline_tumors": {
                "mean": unified_df.groupby("patient_id")["baseline_tumors"].first().mean(),
                "median": unified_df.groupby("patient_id")["baseline_tumors"].first().median(),
            },
            "baseline_size": {
                "mean": unified_df.groupby("patient_id")["baseline_size"].first().mean(),
                "median": unified_df.groupby("patient_id")["baseline_size"].first().median(),
            },
        }

        return summary
