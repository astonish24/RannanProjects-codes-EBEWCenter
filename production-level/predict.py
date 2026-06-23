"""Prediction entrypoint for production-level package."""

from __future__ import annotations

import argparse
import json

from ebewcenter_qa.logging_config import setup_logging
from ebewcenter_qa.service import QAService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QA inference")
    parser.add_argument("--model-id", default="Astonish24/bert-finetuned-squad")
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    service = QAService(model_id=args.model_id)
    result = service.answer(question=args.question, context=args.context)
    print(json.dumps(result.__dict__, ensure_ascii=False))


if __name__ == "__main__":
    main()
