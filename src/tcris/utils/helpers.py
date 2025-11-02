"""Helper utility functions."""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    import yaml

    ensure_dir(config_path.parent)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def train_test_split_stratified(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify_col: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test sets with optional stratification.

    Args:
        df: Input DataFrame
        test_size: Fraction of data for test set
        stratify_col: Column to use for stratification
        random_state: Random seed

    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    if stratify_col:
        stratify = df[stratify_col]
    else:
        stratify = None

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=stratify, random_state=random_state
    )

    logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df


def get_device() -> torch.device:
    """
    Get the best available device (CUDA, MPS, or CPU).

    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
