"""Training entrypoint for production-level package."""

from __future__ import annotations

import argparse
from pathlib import Path

from ebewcenter_qa.config import AppConfig
from ebewcenter_qa.data import load_qa_dataframe, split_dataframe
from ebewcenter_qa.logging_config import setup_logging
from ebewcenter_qa.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train construction QA model")
    parser.add_argument("--data-path", default=None, help="Path to QA CSV")
    parser.add_argument("--model-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def _resolve_data_path(cli_data_path: str | None) -> Path:
    if cli_data_path:
        return Path(cli_data_path)

    root = Path(__file__).resolve().parent.parent
    default_candidates = [
        root / "sample _questions_for_pilot_test.csv",
        root / "sample_questions_for_pilot_test.csv",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return candidate
    return default_candidates[0]


def main() -> None:
    args = parse_args()
    base_config = AppConfig.from_env(default_data_path=_resolve_data_path(args.data_path))
    config = AppConfig(
        data_path=Path(args.data_path) if args.data_path else base_config.data_path,
        model_checkpoint=args.model_checkpoint or base_config.model_checkpoint,
        output_dir=Path(args.output_dir) if args.output_dir else base_config.output_dir,
        batch_size=args.batch_size or base_config.batch_size,
        epochs=args.epochs or base_config.epochs,
        log_level=base_config.log_level,
    )

    setup_logging(config.log_level)
    prepared = load_qa_dataframe(config.data_path)
    train_df, validation_df = split_dataframe(prepared)
    train_model(
        train_df,
        validation_df,
        model_checkpoint=config.model_checkpoint,
        output_dir=str(config.output_dir),
        epochs=config.epochs,
        batch_size=config.batch_size,
    )


if __name__ == "__main__":
    main()
