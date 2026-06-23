"""Model training utilities for extractive QA."""

from __future__ import annotations

from typing import Callable

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from .constants import MAX_LENGTH, STRIDE


def build_dataset_dict(train_df, validation_df) -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(validation_df, preserve_index=False),
        }
    )


def make_preprocess_training_fn(tokenizer) -> Callable:
    def preprocess(examples):
        inputs = tokenizer(
            [q.strip() for q in examples["question"]],
            examples["context"],
            max_length=MAX_LENGTH,
            truncation="only_second",
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offsets = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answer_idx"]
        starts, ends = [], []

        for i, offset in enumerate(offsets):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if start_char < 0 or offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                starts.append(0)
                ends.append(0)
                continue

            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            starts.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            ends.append(idx + 1)

        inputs["start_positions"] = starts
        inputs["end_positions"] = ends
        return inputs

    return preprocess


def train_model(train_df, validation_df, model_checkpoint: str, output_dir: str, epochs: int, batch_size: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    datasets = build_dataset_dict(train_df, validation_df)
    tokenized_train = datasets["train"].map(
        make_preprocess_training_fn(tokenizer),
        batched=True,
        remove_columns=datasets["train"].column_names,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
