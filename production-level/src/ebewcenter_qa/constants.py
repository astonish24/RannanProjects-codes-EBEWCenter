"""Shared constants for training and inference."""

MODEL_CHECKPOINT = "bert-base-cased"
MAX_LENGTH = 384
STRIDE = 128
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 3
DEFAULT_LOG_LEVEL = "INFO"
REQUIRED_COLUMNS = ("question", "answers", "context")
