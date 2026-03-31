"""
train.py
--------
Production training script for BERT-based Extractive Question Answering
on construction course data. Converted faithfully from Bert_QA_base.ipynb.

Pipeline:
  1. Load & clean CSV data
  2. Compute answer start positions (SequenceMatcher)
  3. Assign unique IDs and split 80/20 into train / validation
  4. Tokenise with sliding-window (max_length=384, stride=128)
  5. Fine-tune TFAutoModelForQuestionAnswering with Keras
  6. (Optional) Push to Hugging Face Hub via PushToHubCallback
  7. Run final evaluation — Exact Match + F1

Usage:
    python train.py \
        --data_path  data/sample_questions_for_pilot_test.csv \
        --hub_model_id  your-username/bert-finetuned-squad \
        --push_to_hub \
        --epochs 20 \
        --batch_size 16
"""

import argparse
import collections
import logging
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import Dataset, DatasetDict
from difflib import SequenceMatcher
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    DefaultDataCollator,
    TFAutoModelForQuestionAnswering,
    create_optimizer,
)
from transformers.keras_callbacks import PushToHubCallback

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — match notebook exactly
# ---------------------------------------------------------------------------
MAX_LENGTH = 384
STRIDE = 128
N_BEST = 20
MAX_ANSWER_LENGTH = 30


# ===========================================================================
# 1.  Data loading & answer-position computation  (cells 7 – 10)
# ===========================================================================

def find_answer_start_position(answer: str, context: str) -> dict:
    """
    Locate the character-level start of `answer` inside `context` via
    longest-common-subsequence matching.  Mirrors notebook cell 8.

    Returns {'text': [answer], 'answer_start': [int]}
    answer_start == -1  →  no overlap found.
    """
    match = SequenceMatcher(None, context, answer).find_longest_match(
        0, len(context), 0, len(answer)
    )
    if match.size > 0:
        return {"text": [answer], "answer_start": [match.a]}
    return {"text": [answer], "answer_start": [-1]}


def load_and_prepare_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Read CSV → drop nulls → compute answer_idx → assign unique IDs.
    Mirrors notebook cells 7–9.
    """
    logger.info("Loading data from: %s", csv_path)
    data = pd.read_csv(csv_path)
    raw = pd.DataFrame(data)
    raw.dropna(inplace=True)
    raw.reset_index(drop=True, inplace=True)

    # Cell 9: answer position is computed against the 'question' column
    answer_index = [
        find_answer_start_position(raw.iloc[i]["answers"], raw.iloc[i]["question"])
        for i in range(raw.shape[0])
    ]
    raw["answer_idx"] = answer_index
    raw["id"] = [f"unique-{i}" for i in range(raw.shape[0])]

    logger.info("Prepared %d rows.", len(raw))
    return raw


def build_dataset_dict(df: pd.DataFrame) -> DatasetDict:
    """
    80 / 20 stratified split → HuggingFace DatasetDict.
    Mirrors notebook cell 10.
    """
    train_df = df.sample(frac=0.8, random_state=42)
    validation_df = df.drop(train_df.index)
    logger.info(
        "Split → train: %d | validation: %d", len(train_df), len(validation_df)
    )
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(validation_df, preserve_index=False),
        }
    )


# ===========================================================================
# 2.  Tokenisation  (cells 29, 32)
# ===========================================================================

def make_preprocess_training_fn(tokenizer):
    """
    Returns a batched .map() function for the training split.
    Mirrors notebook cell 29.
    """

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answer_idx"]
        start_positions, end_positions = [], []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Locate context token span
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # Answer outside this window → label (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return preprocess_training_examples


def make_preprocess_validation_fn(tokenizer):
    """
    Returns a batched .map() function for the validation split.
    Mirrors notebook cell 32.
    """

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            # Null-out non-context offsets so they are skipped at prediction time
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    return preprocess_validation_examples


# ===========================================================================
# 3.  Metric computation — EM + F1  (cell 47)
# ===========================================================================

def compute_metrics(start_logits, end_logits, features, examples, metric):
    """
    Convert raw start/end logits to answer spans and score against
    ground truth with the SQuAD metric.
    Mirrors notebook cell 47.
    """
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples, desc="Post-processing"):
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -N_BEST - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -N_BEST - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > MAX_ANSWER_LENGTH
                    ):
                        continue
                    answers.append(
                        {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                    )

        if answers:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    # Cell 44: theoretical_answers uses answer_idx directly
    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answer_idx"]} for ex in examples
    ]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


# ===========================================================================
# 4.  Main training pipeline  (cells 49 – 55)
# ===========================================================================

def train(args):
    # ── Data ─────────────────────────────────────────────────────────────
    df = load_and_prepare_dataframe(args.data_path)
    raw_datasets = build_dataset_dict(df)

    # ── Tokenizer  (cell 17 / 37) ─────────────────────────────────────────
    logger.info("Loading tokenizer: %s", args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    # ── Tokenise  (cells 30, 33) ──────────────────────────────────────────
    logger.info("Tokenising training set…")
    train_dataset = raw_datasets["train"].map(
        make_preprocess_training_fn(tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenising train",
    )

    logger.info("Tokenising validation set…")
    validation_dataset = raw_datasets["validation"].map(
        make_preprocess_validation_fn(tokenizer),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Tokenising validation",
    )

    # ── Model  (cell 49) ─────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model_checkpoint)
    model = TFAutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint)

    # ── TF datasets  (cell 52) ────────────────────────────────────────────
    data_collator = DefaultDataCollator(return_tensors="tf")

    tf_train_dataset = model.prepare_tf_dataset(
        train_dataset,
        collate_fn=data_collator,
        shuffle=True,
        batch_size=args.batch_size,
    )
    tf_eval_dataset = model.prepare_tf_dataset(
        validation_dataset,
        collate_fn=data_collator,
        shuffle=False,
        batch_size=args.batch_size,
    )

    # ── Optimizer + mixed precision  (cell 53) ────────────────────────────
    num_train_steps = len(tf_train_dataset) * args.epochs
    optimizer, _ = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=0,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
    )
    model.compile(optimizer=optimizer)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # ── Callbacks  (cell 54) ─────────────────────────────────────────────
    callbacks = []
    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id is required when --push_to_hub is set.")
        callbacks.append(
            PushToHubCallback(
                output_dir=args.output_dir,
                tokenizer=tokenizer,
                hub_model_id=args.hub_model_id,
            )
        )
        logger.info("PushToHub enabled → %s", args.hub_model_id)

    # ── Train  (cell 54) ─────────────────────────────────────────────────
    logger.info("Training for %d epochs…", args.epochs)
    model.fit(tf_train_dataset, callbacks=callbacks, epochs=args.epochs)

    # ── Save locally ──────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved to: %s", args.output_dir)

    # ── Evaluate  (cell 55) ───────────────────────────────────────────────
    logger.info("Running evaluation…")
    metric = evaluate.load("squad")
    predictions = model.predict(tf_eval_dataset)
    results = compute_metrics(
        predictions["start_logits"],
        predictions["end_logits"],
        validation_dataset,
        raw_datasets["validation"],
        metric,
    )
    logger.info("Results: %s", results)
    return results


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT for extractive QA on construction course data."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/sample_questions_for_pilot_test.csv",
        help="Path to CSV with 'question', 'answers', 'context' columns.",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="bert-base-cased",
        help="HuggingFace model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bert-finetuned-squad",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (notebook default: 20).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (notebook default: 16).",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the trained model to the HuggingFace Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hub repo ID e.g. your-username/bert-finetuned-squad.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train(args)
    print("\n=== Final Evaluation ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
