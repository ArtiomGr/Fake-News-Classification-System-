import pickle
from pathlib import Path

import numpy as np
import torch

from sentiment import analyze_vader_sentiment
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DISTILBERT_DIR = PROJECT_ROOT / "models" / "distilbert"
LOCAL_BERT_DIR = PROJECT_ROOT / "models" / "bert"
HUGGING_FACE_BERT = "Artiomg1/truthlens-bert-base"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.pkl"
TFIDF_VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"

LABELS = {
    0: "True",
    1: "Satire",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content",
}
NUM_LABELS = len(LABELS)

TRANSFORMER_MAX_LENGTH = 128
CHUNK_OVERLAP_TOKENS = 16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading DistilBERT...")
distilbert_tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
distilbert_model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
distilbert_model.to(device)
distilbert_model.eval()
print("DistilBERT loaded successfully.")

if LOCAL_BERT_DIR.exists() and (LOCAL_BERT_DIR / "config.json").exists():
    BERT_SOURCE = str(LOCAL_BERT_DIR)
else:
    BERT_SOURCE = HUGGING_FACE_BERT

print(f"Loading BERT from: {BERT_SOURCE}")
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_SOURCE)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_SOURCE)
bert_model.to(device)
bert_model.eval()
print("BERT loaded successfully.")

print("Loading Logistic Regression...")
with open(BASELINE_MODEL_PATH, "rb") as file:
    logistic_model = pickle.load(file)

with open(TFIDF_VECTORIZER_PATH, "rb") as file:
    tfidf_vectorizer = pickle.load(file)
print("Logistic Regression loaded successfully.")


def _normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).split())


def transformer_predict(text, tokenizer, model, max_length=TRANSFORMER_MAX_LENGTH):
    """Classify a single text chunk using a Transformer model."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)
        probabilities = torch.softmax(output.logits, dim=1)[0]

    return probabilities.cpu().numpy()


def create_chunks(text, tokenizer, max_tokens=TRANSFORMER_MAX_LENGTH, overlap_tokens=CHUNK_OVERLAP_TOKENS):
    """Split text into token-based overlapping chunks.

    The trained Transformer models were configured with a max input length of 128 tokens.
    A chunk may add [CLS] and [SEP] tokens at inference time, so the safe working capacity
    per chunk is 128 - 2 = 126 tokens. We use a small overlap of 16 tokens between chunks
    to preserve boundary context in long documents without creating empty or oversized chunks.
    """
    clean_text = _normalize_text(text)
    if not clean_text:
        return []

    original_model_max_length = getattr(tokenizer, "model_max_length", None)
    tokenizer.model_max_length = 1_000_000
    try:
        token_ids = tokenizer.encode(
            clean_text,
            add_special_tokens=False,
            truncation=False,
            max_length=None,
        )
    finally:
        if original_model_max_length is not None:
            tokenizer.model_max_length = original_model_max_length

    if not token_ids:
        return []

    safe_chunk_size = max(1, max_tokens - 2)
    if safe_chunk_size <= 0:
        return [clean_text]

    if overlap_tokens >= safe_chunk_size:
        overlap_tokens = max(0, safe_chunk_size // 2)

    step = safe_chunk_size - overlap_tokens
    if step <= 0:
        step = safe_chunk_size

    chunks = []
    start = 0

    while start < len(token_ids):
        end = min(start + safe_chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        if not chunk_ids:
            break

        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        chunk_text = " ".join(str(chunk_text).split())
        if chunk_text:
            chunks.append(chunk_text)

        if end == len(token_ids):
            break
        start += step

    return chunks if chunks else [clean_text]


def aggregate_chunk_probabilities(chunk_probabilities):
    """Aggregate chunk predictions by averaging class probabilities."""
    if not chunk_probabilities:
        raise ValueError("At least one chunk is required for aggregation.")

    mean_probabilities = np.mean(np.stack(chunk_probabilities, axis=0), axis=0)
    mean_probabilities = mean_probabilities / np.sum(mean_probabilities)
    return mean_probabilities


def predict_transformer_with_chunks(text, tokenizer, model, max_length=TRANSFORMER_MAX_LENGTH):
    """Predict the document-level class distribution for transformer models with overlapping chunking."""
    clean_text = _normalize_text(text)
    if not clean_text:
        raise ValueError("Input text cannot be empty.")

    chunks = create_chunks(clean_text, tokenizer, max_tokens=max_length, overlap_tokens=CHUNK_OVERLAP_TOKENS)
    if not chunks:
        raise ValueError("Input text could not be converted into valid chunks.")

    chunk_probabilities = []
    for chunk in chunks:
        probabilities = transformer_predict(chunk, tokenizer, model, max_length=max_length)
        chunk_probabilities.append(probabilities)

    average_probabilities = aggregate_chunk_probabilities(chunk_probabilities)
    predicted_class = int(np.argmax(average_probabilities))
    confidence = float(average_probabilities[predicted_class])

    probability_by_label = {
        LABELS[index]: float(average_probabilities[index])
        for index in range(NUM_LABELS)
    }

    return {
        "predicted_class": predicted_class,
        "predicted_label": LABELS[predicted_class],
        "confidence": confidence,
        "probability_by_label": probability_by_label,
        "number_of_chunks": len(chunks),
    }


def predict_logistic_regression(text):
    """Classify text using the TF-IDF + Logistic Regression baseline."""
    clean_text = _normalize_text(text)
    if not clean_text:
        raise ValueError("Input text cannot be empty.")

    features = tfidf_vectorizer.transform([clean_text])
    probabilities = logistic_model.predict_proba(features)[0]
    predicted_class = int(logistic_model.classes_[np.argmax(probabilities)])
    confidence = float(np.max(probabilities))

    probability_by_label = {LABELS[index]: 0.0 for index in range(NUM_LABELS)}
    for class_id, probability in zip(logistic_model.classes_, probabilities):
        probability_by_label[LABELS[int(class_id)]] = float(probability)

    return {
        "predicted_class": predicted_class,
        "predicted_label": LABELS[predicted_class],
        "confidence": confidence,
        "probability_by_label": probability_by_label,
    }


def predict_text(text):
    """Run the same input through all three final project models."""
    clean_text = _normalize_text(text)
    if not clean_text:
        raise ValueError("Input text cannot be empty.")

    bert_result = predict_transformer_with_chunks(clean_text, bert_tokenizer, bert_model)
    distilbert_result = predict_transformer_with_chunks(clean_text, distilbert_tokenizer, distilbert_model)
    logistic_result = predict_logistic_regression(clean_text)

    return {
        "BERT": bert_result,
        "DistilBERT": distilbert_result,
        "Logistic Regression": logistic_result,
    }


def predict_sentiment(text):
    """Run VADER sentiment analysis independently from the fake-news classifiers."""
    return analyze_vader_sentiment(text)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRUTHLENS MULTI-MODEL CLASSIFICATION")
    print("=" * 70)

    while True:
        text = input("\nEnter news text (or type EXIT):\n> ")
        if text.strip().lower() == "exit":
            break

        if not text.strip():
            print("Please enter valid text.")
            continue

        try:
            results = predict_text(text)
            print("\nMODEL COMPARISON")
            print("=" * 70)
            for model_name, result in results.items():
                print(f"\n{model_name}")
                print("-" * 40)
                print("Prediction:", result["predicted_label"])
                print("Confidence:", f"{result['confidence'] * 100:.2f}%")
                if "number_of_chunks" in result:
                    print("Chunks analyzed:", result["number_of_chunks"])

                print("\nClass probabilities:")
                for label, probability in result["probability_by_label"].items():
                    print(f"  {label}: {probability * 100:.2f}%")

            print("\n" + "=" * 70)
        except Exception as error:
            print("Error:", error)