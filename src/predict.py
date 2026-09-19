import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_BERT_DIR = PROJECT_ROOT / "models" / "bert"
LOCAL_DISTILBERT_DIR = PROJECT_ROOT / "models" / "distilbert"

BASELINE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "baseline_model.pkl"
)

TFIDF_VECTORIZER_PATH = (
    PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
)


# ============================================================
# Hugging Face model repositories
# ============================================================

HUGGING_FACE_BERT = (
    "Artiomg1/truthlens-bert-base"
)

HUGGING_FACE_DISTILBERT = (
    "Artiomg1/truthlens-distilbert"
)

# HF_TOKEN is only required for private Hugging Face repositories.
# For public repositories, the value can remain None.
HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# Classification labels
# ============================================================

LABELS = {
    0: "True",
    1: "Satire",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content",
}

NUM_LABELS = len(LABELS)


# ============================================================
# Transformer configuration
# ============================================================

TRANSFORMER_MAX_LENGTH = 128
CHUNK_OVERLAP_TOKENS = 16


# ============================================================
# Device configuration
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# Resolve model source
# ============================================================

def resolve_model_source(
    local_directory,
    huggingface_repository,
):
    """
    Use a local trained model when its config.json exists.
    Otherwise, load the model from Hugging Face.
    """

    config_path = local_directory / "config.json"

    if config_path.is_file():
        return str(local_directory)

    return huggingface_repository


BERT_SOURCE = resolve_model_source(
    LOCAL_BERT_DIR,
    HUGGING_FACE_BERT,
)

DISTILBERT_SOURCE = resolve_model_source(
    LOCAL_DISTILBERT_DIR,
    HUGGING_FACE_DISTILBERT,
)


# ============================================================
# Hugging Face authentication arguments
# ============================================================

def huggingface_arguments():
    """
    Return authentication arguments only when an HF token exists.
    Public repositories do not require a token.
    """

    if HF_TOKEN:
        return {
            "token": HF_TOKEN,
        }

    return {}


# ============================================================
# Model loading
# ============================================================

@lru_cache(maxsize=2)
def load_transformer_model(model_source):
    """
    Load and cache one Transformer tokenizer and model.

    Caching prevents Streamlit from downloading and loading
    the same model again on every application rerun.
    """

    print(f"Loading Transformer model from: {model_source}")

    authentication = huggingface_arguments()

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        **authentication,
    )

    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_source,
            **authentication,
        )
    )

    model.to(device)
    model.eval()

    print(f"Model loaded successfully: {model_source}")

    return tokenizer, model


@lru_cache(maxsize=1)
def load_logistic_regression():
    """
    Load and cache the Logistic Regression model
    and TF-IDF vectorizer.
    """

    if not BASELINE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Logistic Regression model was not found: "
            f"{BASELINE_MODEL_PATH}. "
            "Add models/baseline_model.pkl to the GitHub "
            "repository used by Streamlit."
        )

    if not TFIDF_VECTORIZER_PATH.is_file():
        raise FileNotFoundError(
            "TF-IDF vectorizer was not found: "
            f"{TFIDF_VECTORIZER_PATH}. "
            "Add models/tfidf_vectorizer.pkl to the GitHub "
            "repository used by Streamlit."
        )

    print("Loading Logistic Regression model...")

    with open(BASELINE_MODEL_PATH, "rb") as file:
        logistic_model = pickle.load(file)

    with open(TFIDF_VECTORIZER_PATH, "rb") as file:
        tfidf_vectorizer = pickle.load(file)

    print("Logistic Regression loaded successfully.")

    return logistic_model, tfidf_vectorizer


# ============================================================
# VADER sentiment analyzer
# ============================================================

sentiment_analyzer = SentimentIntensityAnalyzer()


def predict_sentiment(text):
    """
    Analyze the emotional tone of the input using VADER.

    Sentiment is independent from the fake-news category.
    """

    clean_text = _normalize_text(text)

    if not clean_text:
        raise ValueError(
            "Input text cannot be empty."
        )

    scores = sentiment_analyzer.polarity_scores(
        clean_text
    )

    compound = float(scores["compound"])

    if compound >= 0.05:
        sentiment_label = "Positive"
    elif compound <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return {
        "sentiment_label": sentiment_label,
        "compound": compound,
        "positive": float(scores["pos"]),
        "neutral": float(scores["neu"]),
        "negative": float(scores["neg"]),
    }


# ============================================================
# Text normalization
# ============================================================

def _normalize_text(text):
    """
    Convert input to a clean single-space string.
    """

    if text is None:
        return ""

    return " ".join(
        str(text).split()
    )


# ============================================================
# Single-chunk Transformer prediction
# ============================================================

def transformer_predict(
    text,
    tokenizer,
    model,
    max_length=TRANSFORMER_MAX_LENGTH,
):
    """
    Classify one text chunk using a Transformer model.
    """

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        output = model(**encoded)

        probabilities = torch.softmax(
            output.logits,
            dim=1,
        )[0]

    probabilities = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )

    if len(probabilities) != NUM_LABELS:
        raise ValueError(
            "The Transformer model returned "
            f"{len(probabilities)} classes, but "
            f"{NUM_LABELS} classes were expected."
        )

    return probabilities


# ============================================================
# Long-text chunking
# ============================================================

def create_chunks(
    text,
    tokenizer,
    max_tokens=TRANSFORMER_MAX_LENGTH,
    overlap_tokens=CHUNK_OVERLAP_TOKENS,
):
    """
    Split long text into overlapping token-based chunks.

    The models were trained with a maximum input length
    of 128 tokens. Two positions are reserved for special
    tokens, leaving 126 content tokens per chunk.
    """

    clean_text = _normalize_text(text)

    if not clean_text:
        return []

    original_model_max_length = getattr(
        tokenizer,
        "model_max_length",
        None,
    )

    # Prevent a tokenizer warning while creating raw token IDs.
    tokenizer.model_max_length = 1_000_000

    try:
        token_ids = tokenizer.encode(
            clean_text,
            add_special_tokens=False,
            truncation=False,
        )
    finally:
        if original_model_max_length is not None:
            tokenizer.model_max_length = (
                original_model_max_length
            )

    if not token_ids:
        return []

    safe_chunk_size = max(
        1,
        max_tokens - 2,
    )

    overlap_tokens = max(
        0,
        int(overlap_tokens),
    )

    if overlap_tokens >= safe_chunk_size:
        overlap_tokens = safe_chunk_size // 2

    step = safe_chunk_size - overlap_tokens

    if step <= 0:
        step = safe_chunk_size

    chunks = []
    start = 0

    while start < len(token_ids):
        end = min(
            start + safe_chunk_size,
            len(token_ids),
        )

        chunk_ids = token_ids[start:end]

        if not chunk_ids:
            break

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        chunk_text = _normalize_text(
            chunk_text
        )

        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(token_ids):
            break

        start += step

    if chunks:
        return chunks

    return [clean_text]


# ============================================================
# Chunk aggregation
# ============================================================

def aggregate_chunk_probabilities(
    chunk_probabilities,
):
    """
    Calculate document-level probabilities by averaging
    the class probabilities of all chunks.
    """

    if not chunk_probabilities:
        raise ValueError(
            "At least one chunk is required "
            "for aggregation."
        )

    stacked_probabilities = np.stack(
        chunk_probabilities,
        axis=0,
    )

    mean_probabilities = np.mean(
        stacked_probabilities,
        axis=0,
    )

    probability_sum = float(
        np.sum(mean_probabilities)
    )

    if probability_sum <= 0:
        raise ValueError(
            "The model returned an invalid "
            "probability distribution."
        )

    mean_probabilities = (
        mean_probabilities / probability_sum
    )

    return mean_probabilities


# ============================================================
# Full Transformer prediction
# ============================================================

def predict_transformer_with_chunks(
    text,
    tokenizer,
    model,
    max_length=TRANSFORMER_MAX_LENGTH,
):
    """
    Run a Transformer model on all text chunks and return
    one document-level prediction.
    """

    clean_text = _normalize_text(text)

    if not clean_text:
        raise ValueError(
            "Input text cannot be empty."
        )

    chunks = create_chunks(
        clean_text,
        tokenizer,
        max_tokens=max_length,
        overlap_tokens=CHUNK_OVERLAP_TOKENS,
    )

    if not chunks:
        raise ValueError(
            "Input text could not be converted "
            "into valid chunks."
        )

    chunk_probabilities = []

    for chunk in chunks:
        probabilities = transformer_predict(
            chunk,
            tokenizer,
            model,
            max_length=max_length,
        )

        chunk_probabilities.append(
            probabilities
        )

    average_probabilities = (
        aggregate_chunk_probabilities(
            chunk_probabilities
        )
    )

    predicted_class = int(
        np.argmax(average_probabilities)
    )

    confidence = float(
        average_probabilities[predicted_class]
    )

    probability_by_label = {
        LABELS[index]: float(
            average_probabilities[index]
        )
        for index in range(NUM_LABELS)
    }

    return {
        "predicted_class": predicted_class,
        "predicted_label": LABELS[
            predicted_class
        ],
        "confidence": confidence,
        "probability_by_label": (
            probability_by_label
        ),
        "number_of_chunks": len(chunks),
    }


# ============================================================
# Logistic Regression prediction
# ============================================================

def predict_logistic_regression(text):
    """
    Classify text using the trained TF-IDF and
    Logistic Regression baseline.
    """

    clean_text = _normalize_text(text)

    if not clean_text:
        raise ValueError(
            "Input text cannot be empty."
        )

    logistic_model, tfidf_vectorizer = (
        load_logistic_regression()
    )

    features = tfidf_vectorizer.transform(
        [clean_text]
    )

    probabilities = (
        logistic_model.predict_proba(
            features
        )[0]
    )

    highest_probability_index = int(
        np.argmax(probabilities)
    )

    predicted_class = int(
        logistic_model.classes_[
            highest_probability_index
        ]
    )

    if predicted_class not in LABELS:
        raise ValueError(
            "Logistic Regression returned an "
            f"unknown class: {predicted_class}"
        )

    confidence = float(
        probabilities[
            highest_probability_index
        ]
    )

    probability_by_label = {
        LABELS[index]: 0.0
        for index in range(NUM_LABELS)
    }

    for class_id, probability in zip(
        logistic_model.classes_,
        probabilities,
    ):
        class_id = int(class_id)

        if class_id in LABELS:
            probability_by_label[
                LABELS[class_id]
            ] = float(probability)

    return {
        "predicted_class": predicted_class,
        "predicted_label": LABELS[
            predicted_class
        ],
        "confidence": confidence,
        "probability_by_label": (
            probability_by_label
        ),
    }


# ============================================================
# Multi-model prediction
# ============================================================

def predict_text(text):
    """
    Run the same input through BERT, DistilBERT
    and Logistic Regression.
    """

    clean_text = _normalize_text(text)

    if not clean_text:
        raise ValueError(
            "Input text cannot be empty."
        )

    bert_tokenizer, bert_model = (
        load_transformer_model(
            BERT_SOURCE
        )
    )

    (
        distilbert_tokenizer,
        distilbert_model,
    ) = load_transformer_model(
        DISTILBERT_SOURCE
    )

    bert_result = (
        predict_transformer_with_chunks(
            clean_text,
            bert_tokenizer,
            bert_model,
        )
    )

    distilbert_result = (
        predict_transformer_with_chunks(
            clean_text,
            distilbert_tokenizer,
            distilbert_model,
        )
    )

    logistic_result = (
        predict_logistic_regression(
            clean_text
        )
    )

    return {
        "BERT": bert_result,
        "DistilBERT": distilbert_result,
        "Logistic Regression": logistic_result,
    }


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRUTHLENS MULTI-MODEL CLASSIFICATION")
    print("=" * 70)

    while True:
        text = input(
            "\nEnter news text "
            "(or type EXIT):\n> "
        )

        if text.strip().lower() == "exit":
            break

        if not text.strip():
            print("Please enter valid text.")
            continue

        try:
            results = predict_text(text)
            sentiment = predict_sentiment(text)

            print("\nMODEL COMPARISON")
            print("=" * 70)

            for model_name, result in results.items():
                print(f"\n{model_name}")
                print("-" * 40)

                print(
                    "Prediction:",
                    result["predicted_label"],
                )

                print(
                    "Confidence:",
                    f"{result['confidence'] * 100:.2f}%",
                )

                if "number_of_chunks" in result:
                    print(
                        "Chunks analyzed:",
                        result["number_of_chunks"],
                    )

                print("\nClass probabilities:")

                for label, probability in (
                    result[
                        "probability_by_label"
                    ].items()
                ):
                    print(
                        f"  {label}: "
                        f"{probability * 100:.2f}%"
                    )

            print("\nSENTIMENT")
            print("-" * 40)

            print(
                "Label:",
                sentiment["sentiment_label"],
            )

            print(
                "Compound:",
                sentiment["compound"],
            )

            print("\n" + "=" * 70)

        except Exception as error:
            print("Error:", error)