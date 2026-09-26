# TruthLens AI

TruthLens AI is a final engineering project for six-class fake-news and misinformation classification using the Fakeddit dataset.

## Overview

The final active system compares three models on the same input text:

1. BERT
2. DistilBERT
3. TF-IDF + Logistic Regression

The same text is also analyzed independently with VADER for sentiment, not for fake-news classification.

It is designed for the six Fakeddit labels:

- True
- Satire
- False Connection
- Imposter Content
- Manipulated Content
- Misleading Content

## Architecture

Input text
    |
    +--> BERT ----------------------> classification
    +--> DistilBERT ----------------> classification
    +--> TF-IDF + Logistic Regression -> classification
    |
    +--> VADER ---------------------> sentiment analysis

## Model comparison

The project stores controlled evaluation results for the three final fake-news classifiers in the project results folder.

VADER is not included in the model-comparison accuracy table because it is a separate sentiment analysis component, not a fake-news classifier.

Current reported metrics for the final classification comparison are:

- TF-IDF + Logistic Regression: Accuracy 0.6196, Precision 0.7094, Recall 0.6196, F1 0.6467
- DistilBERT: Accuracy 0.7276, Precision 0.7165, Recall 0.7276, F1 0.7183
- BERT: Accuracy 0.7410, Precision 0.7317, Recall 0.7410, F1 0.7329

These are the final comparison results used in the application and should not be replaced with unsupported claims.

## Long-text handling and chunking

The BERT and DistilBERT models are run on long input using token-based chunking instead of character splitting.

Implementation details:

- The Transformer models use a 128-token max sequence length.
- Chunks are generated using the tokenizer directly, not raw string splitting.
- Special tokens are accounted for by leaving room inside the 128-token limit.
- A small overlap is used between adjacent chunks to preserve context at boundaries.
- Each chunk is classified independently.
- Final document-level probabilities are created by averaging the chunk-level class probabilities.
- Short text remains a single chunk.

This preserves the trained model configuration while allowing the system to process long headlines or articles.

## Project structure

- app/app.py — Streamlit application
- src/predict.py — final active prediction interface for all three models
- src/baseline.py — TF-IDF + Logistic Regression training/evaluation
- src/train_bert.py — BERT training experiment
- src/train_distilbert.py — DistilBERT training experiment
- src/compare_models.py — final comparison script for the three active project models
- models/ — trained model files and vectorizer
- results/ — evaluation metrics and comparison outputs
- data/ — Fakeddit train/validate/test files

## Installation

Create a virtual environment and install requirements:

```bash
python -m venv .venv
. .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate # Windows
pip install -r requirements.txt
```

## Running the app

From the project root:

```bash
streamlit run app/app.py
```

The app reads the same user input and runs:

- BERT
- DistilBERT
- TF-IDF + Logistic Regression

It then separately runs VADER sentiment analysis on the same input and displays the sentiment output independently from the fake-news classification results.

## Model files and outputs

Expected active artifacts:

- models/bert/
- models/distilbert/
- models/baseline_model.pkl
- models/tfidf_vectorizer.pkl
- results/baseline_metrics.csv
- results/distilbert_metrics.csv
- results/BERT_metrics.csv
- results/model_comparison.csv

Legacy or research artifacts for earlier experiments or sentiment fusion remain in the repository for traceability but are not part of the final active application.

## Notes

- VADER is used as a separate sentiment-analysis component and is not part of the fake-news classification stack.
- Confidence values are model probabilities and should not be treated as independent fact verification.
- The final active fake-news classification stack is BERT, DistilBERT, and TF-IDF + Logistic Regression only.
