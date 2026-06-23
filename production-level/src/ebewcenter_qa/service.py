"""Inference service layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QAResult:
    answer: str
    score: float
    start: int
    end: int


class QAService:
    """Thin service wrapper around transformers QA pipeline."""

    def __init__(self, model_id: str):
        from transformers import pipeline

        self._pipeline = pipeline("question-answering", model=model_id)

    def answer(self, question: str, context: str) -> QAResult:
        result = self._pipeline(question=question, context=context)
        return QAResult(
            answer=result.get("answer", ""),
            score=float(result.get("score", 0.0)),
            start=int(result.get("start", 0)),
            end=int(result.get("end", 0)),
        )
