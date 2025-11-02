#!/usr/bin/env python3
"""
Quick demonstration of T-CRIS data loading and fusion capabilities.
Run this to verify the installation and see the core functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from tcris.data.loaders import BladderDataLoader
from tcris.data.fusion import DataFusionEngine


def main():
    """Run quick demonstration."""
    logger.info("=" * 70)
    logger.info("T-CRIS Quick Demonstration")
    logger.info("=" * 70)

    # Step 1: Load data
    logger.info("\n[Step 1] Loading bladder cancer datasets...")
    loader = BladderDataLoader()

    try:
        datasets = loader.load_all()
        logger.info(f"✓ Successfully loaded {len(datasets)} datasets")

        for filename, (df, format) in datasets.items():
            summary = loader.get_dataset_summary(df)
            logger.info(f"\n{filename}:")
            logger.info(f"  - Format: {format}")
            logger.info(f"  - Shape: {summary['n_rows']} rows × {summary['n_columns']} columns")
            logger.info(f"  - Patients: {summary['n_patients']}")
            if "treatment_distribution" in summary:
                logger.info(f"  - Treatments: {summary['treatment_distribution']}")

    except Exception as e:
        logger.error(f"✗ Failed to load datasets: {e}")
        logger.error("Make sure CSV files are in data/ directory")
        return 1

    # Step 2: Fuse data
    logger.info("\n[Step 2] Fusing datasets into unified format...")
    try:
        fusion_engine = DataFusionEngine()
        unified_df = fusion_engine.fuse(datasets)

        logger.info(f"✓ Successfully fused datasets")
        logger.info(f"  - Total rows: {len(unified_df)}")
        logger.info(f"  - Unique patients: {unified_df['patient_id'].nunique()}")

        # Get summary
        summary = fusion_engine.summarize_unified_data(unified_df)
        logger.info(f"\n  Unified Data Summary:")
        logger.info(f"  - Treatment distribution: {summary['treatment_distribution']}")
        logger.info(f"  - Event types: {summary['event_type_distribution']}")
        logger.info(f"  - Mean follow-up time: {summary['follow_up_time']['mean']:.1f} months")
        logger.info(f"  - Mean baseline tumors: {summary['baseline_tumors']['mean']:.1f}")
        logger.info(f"  - Mean baseline size: {summary['baseline_size']['mean']:.1f} cm")

    except Exception as e:
        logger.error(f"✗ Failed to fuse datasets: {e}")
        return 1

    # Step 3: Show example patient trajectory
    logger.info("\n[Step 3] Example patient trajectory...")
    try:
        patient_id = unified_df["patient_id"].iloc[0]
        trajectory = fusion_engine.get_patient_trajectory(unified_df, patient_id)

        logger.info(f"\nPatient {patient_id} trajectory:")
        logger.info(f"  - Treatment: {trajectory['treatment'].iloc[0]}")
        logger.info(f"  - Baseline tumors: {trajectory['baseline_tumors'].iloc[0]}")
        logger.info(f"  - Baseline size: {trajectory['baseline_size'].iloc[0]} cm")
        logger.info(f"  - Number of events: {len(trajectory)}")

        logger.info(f"\n  Event timeline:")
        for idx, row in trajectory.iterrows():
            event_type = {0: "Censored", 1: "Recurrence", 2: "Death (bladder)", 3: "Death (other)"}.get(
                row["event_type"], "Unknown"
            )
            logger.info(
                f"    Event {row['event_number']}: "
                f"[{row['start_time']:.1f} - {row['stop_time']:.1f}] months → {event_type}"
            )

    except Exception as e:
        logger.error(f"✗ Failed to extract patient trajectory: {e}")
        return 1

    # Step 4: Recurrence statistics
    logger.info("\n[Step 4] Recurrence statistics...")
    try:
        recurrence_counts = fusion_engine.get_recurrence_counts(unified_df)

        logger.info(f"  - Patients with 0 recurrences: {(recurrence_counts == 0).sum()}")
        logger.info(f"  - Patients with 1+ recurrences: {(recurrence_counts >= 1).sum()}")
        logger.info(f"  - Mean recurrences: {recurrence_counts.mean():.2f}")
        logger.info(f"  - Max recurrences: {recurrence_counts.max()}")

    except Exception as e:
        logger.error(f"✗ Failed to compute recurrence statistics: {e}")
        return 1

    logger.info("\n" + "=" * 70)
    logger.info("✓ Demo completed successfully!")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("  1. Run 'make train' to train models")
    logger.info("  2. Run 'make dashboard' to launch interactive dashboard")
    logger.info("  3. Run 'make api' to start REST API server")
    logger.info("  4. Explore notebooks/ directory for detailed analysis")
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
