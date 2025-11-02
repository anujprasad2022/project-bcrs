#!/usr/bin/env python3
"""
Train all models and save results.
This script implements and trains all models in one go for rapid development.
"""

import sys
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import torch
import torch.nn as nn
from loguru import logger

from tcris.data.loaders import BladderDataLoader
from tcris.data.fusion import DataFusionEngine
from tcris.features.extractors import create_features


def prepare_data():
    """Load and prepare data."""
    logger.info("Loading data...")
    loader = BladderDataLoader()
    datasets = loader.load_all()

    logger.info("Fusing datasets...")
    fusion = DataFusionEngine()
    unified_df = fusion.fuse(datasets)

    logger.info("Creating features...")
    df_features = create_features(unified_df)

    # Prepare for modeling - use first event per patient
    patient_data = df_features.groupby("patient_id").first().reset_index()

    # Features for modeling
    feature_cols = [
        "baseline_tumors", "baseline_size",
        "tumor_burden_index", "baseline_burden",
        "time_to_first_recurrence", "recurrence_rate",
        "treat_placebo", "treat_thiotepa"
    ]

    # Handle missing treatment columns
    for col in ["treat_placebo", "treat_thiotepa"]:
        if col not in patient_data.columns:
            patient_data[col] = 0

    X = patient_data[feature_cols].fillna(0)
    y_time = patient_data["stop_time"].values
    y_event = (patient_data["event_type"] == 1).astype(int).values

    # Train/test split
    X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    logger.info(f"Train: {len(X_train)} patients, Test: {len(X_test)} patients")

    return (X_train_scaled, X_test_scaled, y_time_train, y_time_test,
            y_event_train, y_event_test, scaler, feature_cols)


def train_cox_model(X_train, y_time_train, y_event_train):
    """Train Cox PH model."""
    logger.info("Training Cox PH model...")

    # Prepare data for lifelines
    df_train = X_train.copy()
    df_train["duration"] = y_time_train
    df_train["event"] = y_event_train

    # Fit model
    cox = CoxPHFitter(penalizer=0.01)
    cox.fit(df_train, duration_col="duration", event_col="event", show_progress=False)

    logger.info(f"Cox model C-index: {cox.concordance_index_:.3f}")

    return cox


def train_rsf_model(X_train, y_time_train, y_event_train):
    """Train Random Survival Forest."""
    logger.info("Training Random Survival Forest...")

    # Create structured array for sksurv
    y_train = Surv.from_arrays(y_event_train.astype(bool), y_time_train)

    # Train RSF
    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    rsf.fit(X_train.values, y_train)

    logger.info("RSF model trained successfully")

    return rsf


class SimpleLSTM(nn.Module):
    """Simple LSTM for survival prediction."""

    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features) - but we treat each sample as sequence of 1
        x = x.unsqueeze(1)  # Add sequence dimension
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()


def train_lstm_model(X_train, y_time_train, y_event_train):
    """Train simple LSTM model."""
    logger.info("Training LSTM model...")

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train.values)
    y_time_tensor = torch.FloatTensor(y_time_train)
    y_event_tensor = torch.FloatTensor(y_event_train)

    # Create model
    model = SimpleLSTM(input_size=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Quick training (10 epochs)
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        # Predict log-time (helps with scale)
        loss = criterion(predictions, torch.log(y_time_tensor + 1))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            logger.debug(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    logger.info("LSTM model trained successfully")

    return model


def evaluate_models(models, X_test, y_time_test, y_event_test):
    """Evaluate all models."""
    from sksurv.metrics import concordance_index_censored

    results = {}

    # Cox model
    if "cox" in models:
        cox = models["cox"]
        risk_scores = cox.predict_partial_hazard(X_test).values
        c_index = concordance_index_censored(
            y_event_test.astype(bool), y_time_test, risk_scores
        )[0]
        results["cox"] = {"c_index": c_index}
        logger.info(f"Cox C-index: {c_index:.3f}")

    # RSF model
    if "rsf" in models:
        rsf = models["rsf"]
        risk_scores = rsf.predict(X_test.values)
        c_index = concordance_index_censored(
            y_event_test.astype(bool), y_time_test, -risk_scores  # RSF returns survival, we want risk
        )[0]
        results["rsf"] = {"c_index": c_index}
        logger.info(f"RSF C-index: {c_index:.3f}")

    # LSTM model
    if "lstm" in models:
        lstm = models["lstm"]
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test.values)
            predictions = lstm(X_test_tensor).numpy()
        c_index = concordance_index_censored(
            y_event_test.astype(bool), y_time_test, -predictions  # Negative for risk
        )[0]
        results["lstm"] = {"c_index": c_index}
        logger.info(f"LSTM C-index: {c_index:.3f}")

    # Ensemble (simple average of risk scores)
    if len(models) > 1:
        ensemble_risk = np.zeros(len(X_test))
        if "cox" in models:
            ensemble_risk += cox.predict_partial_hazard(X_test).values / len(models)
        if "rsf" in models:
            ensemble_risk += -rsf.predict(X_test.values) / len(models)
        if "lstm" in models:
            with torch.no_grad():
                ensemble_risk += -lstm(X_test_tensor).numpy() / len(models)

        c_index = concordance_index_censored(
            y_event_test.astype(bool), y_time_test, ensemble_risk
        )[0]
        results["ensemble"] = {"c_index": c_index}
        logger.info(f"Ensemble C-index: {c_index:.3f}")

    return results


def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("T-CRIS Model Training Pipeline")
    logger.info("=" * 70)

    # Prepare data
    (X_train, X_test, y_time_train, y_time_test,
     y_event_train, y_event_test, scaler, feature_cols) = prepare_data()

    # Train models
    models = {}

    try:
        models["cox"] = train_cox_model(X_train, y_time_train, y_event_train)
    except Exception as e:
        logger.error(f"Cox model failed: {e}")

    try:
        models["rsf"] = train_rsf_model(X_train, y_time_train, y_event_train)
    except Exception as e:
        logger.error(f"RSF model failed: {e}")

    try:
        models["lstm"] = train_lstm_model(X_train, y_time_train, y_event_train)
    except Exception as e:
        logger.error(f"LSTM model failed: {e}")

    # Evaluate models
    logger.info("\n" + "=" * 70)
    logger.info("Model Evaluation")
    logger.info("=" * 70)

    results = evaluate_models(models, X_test, y_time_test, y_event_test)

    # Save models and results
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    if "cox" in models:
        with open(output_dir / "cox_model.pkl", "wb") as f:
            pickle.dump(models["cox"], f)

    if "rsf" in models:
        with open(output_dir / "rsf_model.pkl", "wb") as f:
            pickle.dump(models["rsf"], f)

    if "lstm" in models:
        torch.save(models["lstm"].state_dict(), output_dir / "lstm_model.pt")

    # Save scaler and feature names
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(output_dir / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nModels saved to {output_dir}/")
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    main()
