# Production-Level QA Codebase

This folder contains a production-oriented Python implementation of the Construction QA workflow used in this repository.

## Structure

```text
production-level/
├── predict.py
├── train.py
├── requirements.txt
├── src/ebewcenter_qa/
│   ├── config.py
│   ├── constants.py
│   ├── data.py
│   ├── exceptions.py
│   ├── logging_config.py
│   ├── service.py
│   └── training.py
└── tests/
    └── test_data.py
```

## What is included

- Config/constants via `AppConfig` + `constants.py`
- Structured logging setup (`logging_config.py`)
- Centralized errors (`exceptions.py`)
- Data loading + schema validation + answer index preparation (`data.py`)
- Training utility and CLI entrypoint (`training.py`, `train.py`)
- Inference service and CLI entrypoint (`service.py`, `predict.py`)
- Reproducible dependencies (`requirements.txt`)

## Install

```bash
cd production-level
python -m pip install -r requirements.txt
```

## Train

```bash
cd production-level
PYTHONPATH=src python train.py --data-path ../sample\ _questions_for_pilot_test.csv --output-dir artifacts/model
```

## Predict

```bash
cd production-level
PYTHONPATH=src python predict.py \
  --model-id Astonish24/bert-finetuned-squad \
  --question "What is PPE?" \
  --context "PPE means personal protective equipment in construction."
```

## Test

```bash
cd production-level
pytest -q
```
