"""
Application settings using Pydantic Settings for type-safe configuration.
Single source of truth for all configuration (DRY principle).
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Environment
    environment: Literal["development", "production", "testing"] = "development"
    debug: bool = True

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = Field(default_factory=lambda: Path("data/processed"))
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    outputs_dir: Path = Field(default_factory=lambda: Path("outputs"))

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # Model Settings
    default_model: str = "ensemble"
    random_seed: int = 42

    # Training Settings
    n_folds: int = 5
    test_size: float = 0.2
    batch_size: int = 32
    max_epochs: int = 100
    learning_rate: float = 0.001

    # Dashboard Settings
    dashboard_port: int = 8501
    cache_ttl: int = 3600

    # Logging
    log_level: str = "INFO"
    log_file: str = "tcris.log"

    def get_absolute_path(self, relative_path: Path) -> Path:
        """Convert relative path to absolute path from project root."""
        if relative_path.is_absolute():
            return relative_path
        return self.project_root / relative_path

    @property
    def data_path(self) -> Path:
        """Absolute path to data directory."""
        return self.get_absolute_path(self.data_dir)

    @property
    def processed_data_path(self) -> Path:
        """Absolute path to processed data directory."""
        return self.get_absolute_path(self.processed_data_dir)

    @property
    def models_path(self) -> Path:
        """Absolute path to models directory."""
        return self.get_absolute_path(self.models_dir)

    @property
    def outputs_path(self) -> Path:
        """Absolute path to outputs directory."""
        return self.get_absolute_path(self.outputs_dir)


# Singleton instance
settings = Settings()
