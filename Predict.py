"""
predict.py
----------
Run inference with the fine-tuned BERT construction QA model.
Mirrors notebook cells 56 – 60.

Three modes:

  1. Single question via CLI flags
     python predict.py \
         --model_id Astonish24/bert-finetuned-squad \
         --question "What materials are used for an uncovered deck?" \
         --context  "An uncovered deck is a flat, roofless platform..."

  2. Batch CSV inference
     python predict.py \
         --model_id Astonish24/bert-finetuned-squad \
         --input_csv  data/sample_questions_for_pilot_test.csv \
         --output_csv results/predictions.csv

  3. Interactive REPL (no extra flags needed)
     python predict.py --model_id Astonish24/bert-finetuned-squad
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from transformers import pipeline

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Core predictor  (mirrors notebook cells 56 – 60)
# ===========================================================================

class ConstructionQAPredictor:
    """
    Thin wrapper around the HuggingFace `question-answering` pipeline.
    Matches the notebook's usage pattern exactly.
    """

    def __init__(self, model_id: str):
        logger.info("Loading pipeline from: %s", model_id)
        # Mirrors cells 56 / 58:
        #   question_answerer = pipeline("question-answering", model=model_checkpoint)
        self.pipe = pipeline("question-answering", model=model_id)
        logger.info("Pipeline ready.")

    def predict(self, question: str, context: str) -> dict:
        """
        Answer a single question.

        Returns the raw dict from the pipeline, which includes:
            score   : float  — model confidence
            start   : int    — char offset in context
            end     : int    — char offset in context
            answer  : str    — extracted answer span
        """
        # Mirrors notebook cells 59 / 60
        return self.pipe(question=question, context=context)

    def predict_batch(
        self, questions: list, contexts: list
    ) -> list:
        """Run predict() over parallel lists of questions and contexts."""
        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have the same length.")
        return [self.predict(q, c) for q, c in zip(questions, contexts)]


# ===========================================================================
# Batch CSV inference
# ===========================================================================

def run_batch_inference(
    predictor: ConstructionQAPredictor,
    input_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    Read a CSV with 'question' and 'context' columns, run inference,
    and write results to output_csv.
    """
    df = pd.read_csv(input_csv).dropna()
    logger.info("Running inference on %d rows…", len(df))

    results = predictor.predict_batch(
        df["question"].tolist(), df["context"].tolist()
    )

    df["predicted_answer"] = [r["answer"] for r in results]
    df["prediction_score"] = [round(r["score"], 4) for r in results]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Saved predictions to: %s", output_csv)
    return df


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for the BERT construction QA model."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Astonish24/bert-finetuned-squad",
        help="HuggingFace Hub model ID or local path.",
    )
    # Single-question mode
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--context",  type=str, default=None)
    # Batch mode
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="CSV with 'question' and 'context' columns.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/predictions.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = ConstructionQAPredictor(args.model_id)

    if args.question and args.context:
        # ── Single question ──────────────────────────────────────────────
        result = predictor.predict(args.question, args.context)
        print(f"\nQuestion : {args.question}")
        print(f"Answer   : {result['answer']}")
        print(f"Score    : {result['score']:.4f}")

    elif args.input_csv:
        # ── Batch CSV ────────────────────────────────────────────────────
        df = run_batch_inference(predictor, args.input_csv, args.output_csv)
        cols = ["question", "predicted_answer", "prediction_score"]
        if "answers" in df.columns:
            cols.insert(1, "answers")
        print(df[cols].to_string(index=False))

    else:
        # ── Interactive REPL (mirrors notebook cells 59 / 60) ────────────
        print("Construction QA — Interactive Mode  (type 'quit' to exit)\n")
        print("Tip: paste a long passage as context, then ask your question.\n")
        while True:
            question = input("Question : ").strip()
            if question.lower() in {"quit", "exit", "q"}:
                break
            context = input("Context  : ").strip()
            result = predictor.predict(question, context)
            print(f"Answer   : {result['answer']}  (score: {result['score']:.4f})\n")
