"""Runtime configuration handling."""

from dataclasses import dataclass
from pathlib import Path
import os

from .constants import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LOG_LEVEL, MODEL_CHECKPOINT


@dataclass(frozen=True)
class AppConfig:
    """Configuration for train/predict entrypoints."""

    data_path: Path
    model_checkpoint: str = MODEL_CHECKPOINT
    output_dir: Path = Path("artifacts/model")
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    log_level: str = DEFAULT_LOG_LEVEL

    @staticmethod
    def from_env(default_data_path: Path) -> "AppConfig":
        """Build config from environment variables with sensible defaults."""
        return AppConfig(
            data_path=Path(os.getenv("QA_DATA_PATH", str(default_data_path))),
            model_checkpoint=os.getenv("QA_MODEL_CHECKPOINT", MODEL_CHECKPOINT),
            output_dir=Path(os.getenv("QA_OUTPUT_DIR", "artifacts/model")),
            batch_size=int(os.getenv("QA_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
            epochs=int(os.getenv("QA_EPOCHS", str(DEFAULT_EPOCHS))),
            log_level=os.getenv("QA_LOG_LEVEL", DEFAULT_LOG_LEVEL),
        )
