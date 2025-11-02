#!/usr/bin/env python3
"""
Verify the complete system is working.
Run this script to test all components before the presentation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Test data loading."""
    logger.info("Testing data loading...")
    from tcris.data.loaders import BladderDataLoader
    from tcris.data.fusion import DataFusionEngine

    try:
        loader = BladderDataLoader()
        datasets = loader.load_all()
        assert len(datasets) == 3, "Should load 3 datasets"

        fusion = DataFusionEngine()
        unified_df = fusion.fuse(datasets)
        assert len(unified_df) > 0, "Unified data should not be empty"
        assert unified_df["patient_id"].nunique() == 118, "Should have 118 patients"

        logger.info("✓ Data loading: PASSED")
        return True
    except Exception as e:
        logger.error(f"✗ Data loading: FAILED - {e}")
        return False

def test_feature_engineering():
    """Test feature engineering."""
    logger.info("Testing feature engineering...")
    from tcris.data.loaders import BladderDataLoader
    from tcris.data.fusion import DataFusionEngine
    from tcris.features.extractors import create_features

    try:
        loader = BladderDataLoader()
        datasets = loader.load_all()
        fusion = DataFusionEngine()
        unified_df = fusion.fuse(datasets)

        df_features = create_features(unified_df)
        assert len(df_features.columns) > 20, "Should have 20+ features"
        assert "tumor_burden_index" in df_features.columns, "Should have tumor burden feature"

        logger.info("✓ Feature engineering: PASSED")
        return True
    except Exception as e:
        logger.error(f"✗ Feature engineering: FAILED - {e}")
        return False

def test_models_exist():
    """Test that trained models exist."""
    logger.info("Testing model files...")
    model_dir = Path("models")

    required_files = [
        "cox_model.pkl",
        "rsf_model.pkl",
        "lstm_model.pt",
        "scaler.pkl",
        "results.json"
    ]

    all_exist = True
    for filename in required_files:
        filepath = model_dir / filename
        if filepath.exists():
            logger.info(f"  ✓ {filename} exists")
        else:
            logger.warning(f"  ✗ {filename} NOT FOUND")
            all_exist = False

    if all_exist:
        logger.info("✓ Model files: PASSED")
    else:
        logger.warning("✗ Some model files missing - run train_all_models.py")

    return all_exist

def test_dashboard_imports():
    """Test dashboard can import required modules."""
    logger.info("Testing dashboard imports...")

    try:
        import streamlit
        import plotly
        logger.info("  ✓ Streamlit and Plotly available")

        # Test critical imports from dashboard
        from tcris.data.loaders import BladderDataLoader
        from tcris.data.fusion import DataFusionEngine
        from tcris.features.extractors import create_features
        logger.info("  ✓ All dashboard imports work")

        logger.info("✓ Dashboard imports: PASSED")
        return True
    except Exception as e:
        logger.error(f"✗ Dashboard imports: FAILED - {e}")
        return False

def test_predictions():
    """Test prediction functionality."""
    logger.info("Testing predictions...")

    try:
        import pickle
        import pandas as pd

        # Load Cox model
        with open("models/cox_model.pkl", "rb") as f:
            cox = pickle.load(f)

        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Create test input
        test_input = pd.DataFrame({
            "baseline_tumors": [3],
            "baseline_size": [2.5],
            "tumor_burden_index": [7.5],
            "baseline_burden": [7.5],
            "time_to_first_recurrence": [24],
            "recurrence_rate": [0.1],
            "treat_placebo": [1],
            "treat_thiotepa": [0]
        })

        test_scaled = pd.DataFrame(
            scaler.transform(test_input),
            columns=test_input.columns
        )

        # Get prediction
        risk = cox.predict_partial_hazard(test_scaled).values[0]
        surv = cox.predict_survival_function(test_scaled, times=[24])

        assert risk > 0, "Risk score should be positive"
        assert 0 <= surv.iloc[0, 0] <= 1, "Survival prob should be in [0,1]"

        logger.info(f"  Test prediction: Risk={risk:.3f}, 24-mo survival={surv.iloc[0, 0]:.3f}")
        logger.info("✓ Predictions: PASSED")
        return True
    except Exception as e:
        logger.error(f"✗ Predictions: FAILED - {e}")
        return False

def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("T-CRIS System Verification")
    logger.info("=" * 70)

    tests = [
        ("Data Loading", test_data_loading),
        ("Feature Engineering", test_feature_engineering),
        ("Model Files", test_models_exist),
        ("Dashboard Imports", test_dashboard_imports),
        ("Predictions", test_predictions),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n[{test_name}]")
        result = test_func()
        results.append((test_name, result))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:.<40} {status}")

    logger.info("=" * 70)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! System is ready for presentation!")
        logger.info("\nTo launch dashboard:")
        logger.info("  python3 -m streamlit run dashboard/app.py")
        return 0
    else:
        logger.warning("\n⚠️  Some tests failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
