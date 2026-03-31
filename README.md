# RannanProjects-codes-EBEWCenter
EBEW Center NCaT

# 🏗️ Construction Course Language Model for Question-Answering (QA)

A **Transformer-based Question-Answering system** fine-tuned on construction course materials, enabling students to ask natural language questions and receive accurate, contextually relevant answers on topics such as safety protocols, materials, methods, and construction best practices.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Framework](#framework)
- [Model Architecture](#model-architecture)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Challenges & Future Work](#challenges--future-work)

---

## Project Overview

This project develops a domain-specific QA system for construction education. Students can submit queries in natural language and receive answers grounded in a curated knowledge base of construction textbooks, industry guidelines, and standards.

**Example queries the system can handle:**
- *"What are the key safety protocols for high-rise construction?"*
- *"What materials are required for concrete slab construction?"*
- *"How do I estimate the cost of materials for a bridge construction?"*
- *"Why is it important to avoid land identified as critical for agriculture in the Sensitive Land Protection credit?"*

---

## Framework

The system follows a four-stage pipeline:

```
Domain-Specific Data Sources
        │
        ▼
┌─────────────────────────┐
│  Data Acquisition &     │  Tokenization, Cleaning,
│  Preprocessing          │  Formatting, Train/Val/Test Split
└─────────┬───────────────┘
          │  Preprocessed Data
          ▼
┌─────────────────────────┐
│  Model Selection &      │  BERT (Extractive QA) / GPT (Generative QA)
│  Training               │  Supervised Learning, EM & F1 Evaluation
└─────────┬───────────────┘
          │  Trained Model
          ▼
┌─────────────────────────┐
│  System Deployment &    │  Web Interface, Real-time API Calls,
│  Interaction            │  Scalability & Caching
└─────────┬───────────────┘
          │  User Interaction Data
          ▼
┌─────────────────────────┐
│  Feedback Loop &        │  Capture Low-Performance Queries,
│  Continuous Enhancement │  Expert Review, Retraining
└─────────────────────────┘
```

See [`Framework_Diagram.pdf`](./Framework_Diagram.pdf) for the full visual diagram.

---

## Model Architecture

| Model | Task | Description |
|---|---|---|
| **BERT** (`bert-base-cased`) | Extractive QA | Identifies and extracts answer spans directly from the provided context |
| **DistilBERT** (`distilbert-base-cased-distilled-squad`) | Extractive QA (lightweight) | Faster inference, used for evaluation baseline |
| **GPT** | Generative QA | Generates free-form answers when extractive span selection is insufficient |

The primary notebook (`Bert_QA_base.ipynb`) focuses on **BERT fine-tuning** for extractive QA using the Hugging Face `transformers` library.

---

## Repository Structure

```
construction-qa-llm/
│
├── notebooks/
│   └── Bert_QA_base.ipynb          # Main training & evaluation notebook
│
├── data/
│   └── sample_questions_for_pilot_test.csv   # Sample QA pairs with context
│
├── docs/
│   ├── Project_Overview.pdf        # Full project description
│   └── Framework_Diagram.pdf       # System architecture diagram
│
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset (`sample_questions_for_pilot_test.csv`) consists of construction course QA pairs with the following schema:

| Column | Description |
|---|---|
| `question` | Natural language question from a student |
| `answers` | Ground truth answer text |
| `context` | Relevant passage from construction course materials |

**Data sources include:**
- Construction course textbooks
- LEED (Leadership in Energy and Environmental Design) guidelines
- Industry standards and safety manuals

**Preprocessing steps applied:**
1. Load CSV and drop null rows
2. Locate answer start positions within context using `SequenceMatcher`
3. Format into `{'text': [...], 'answer_start': [...]}` for SQuAD-style training
4. Convert to Hugging Face `Dataset` format with train/validation splits
5. Tokenize with sliding window (stride) to handle long contexts

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/construction-qa-llm.git
cd construction-qa-llm

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
transformers[sentencepiece]>=4.31.0
datasets>=3.2.0
evaluate>=0.4.3
torch
tensorflow
numpy
pandas
huggingface_hub
```

> **Note:** A GPU environment is strongly recommended for fine-tuning. The notebook was developed using Python 3.9.

---

## Usage

### Running the Notebook

Open and run `notebooks/Bert_QA_base.ipynb` step by step:

```bash
jupyter notebook notebooks/Bert_QA_base.ipynb
```

**Before running**, update the git config cell with your own credentials:

```python
!git config --global user.email "your-email@example.com"
!git config --global user.name "YourUsername"
```

And log in to Hugging Face Hub when prompted:

```python
from huggingface_hub import notebook_login
notebook_login()
```

### Quick Inference (after training)

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="<your-hf-username>/construction-qa-bert")

context = """
The Sensitive Land Protection credit awards 1 to 2 points to projects that avoid 
environmentally sensitive lands, minimizing ecological impact of development...
"""
question = "How many points does the Sensitive Land Protection credit award?"

result = qa_pipeline(question=question, context=context)
print(result["answer"])
```

---

## Training Pipeline

The notebook implements the following pipeline:

### 1. Data Loading & Formatting
```python
data = pd.read_csv('sample_questions_for_pilot_test.csv')
# Answer start positions are located using SequenceMatcher
# Data is converted to HuggingFace DatasetDict with train/validation splits
```

### 2. Tokenization with Sliding Window
```python
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# Handles long contexts via stride/overflow tokens
```

### 3. Model Fine-tuning
```python
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)
trainer.train()
```

### 4. Model Push to Hub
```python
trainer.push_to_hub()
```

---

## Evaluation

The model is evaluated using standard QA metrics:

| Metric | Description |
|---|---|
| **Exact Match (EM)** | % of predictions that exactly match the ground truth answer |
| **F1 Score** | Harmonic mean of token-level precision and recall |

These are computed using the Hugging Face `evaluate` library with the `squad` metric.

---

## Deployment

The trained model is intended for integration into a **web-based interface** where:

- Students submit questions via a text input
- The system retrieves the relevant context from the knowledge base
- The model generates or extracts an answer in real time
- User interactions (queries, feedback) are logged for continuous improvement

The feedback loop captures low-confidence responses for expert review and retraining, ensuring the system improves over time as new course materials are added.

---

## Challenges & Future Work

- **Out-of-scope questions:** The model may not always have sufficient context for highly specialized queries
- **Multi-part questions:** Complex questions spanning multiple topics remain challenging for extractive models
- **Knowledge base expansion:** Continuous addition of new construction standards and textbooks
- **GPT integration:** Extending the pipeline to support generative answers for open-ended queries
- **Web application:** Deploying the model behind a scalable REST API with a student-facing UI

---

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [SQuAD: The Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- LEED Reference Guide for Building Design and Construction
