import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# Project and model paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "distilbert"

HUGGING_FACE_MODEL = "Artiomg1/truthlens-bert-base"

# Use the local model when it exists.
# Otherwise, download the model from Hugging Face.
MODEL_SOURCE = (
    str(LOCAL_MODEL_DIR)
    if LOCAL_MODEL_DIR.exists()
    else HUGGING_FACE_MODEL
)


# ============================================================
# Classification labels
# ============================================================

LABELS = {
    0: "True",
    1: "Satire",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content"
}


# ============================================================
# Device configuration
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Loading model from: {MODEL_SOURCE}")
print(f"Loading model on device: {device}")


# ============================================================
# Load tokenizer and trained model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_SOURCE
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_SOURCE
)

model.to(device)
model.eval()


# ============================================================
# Sentiment analyzer
# ============================================================

sentiment_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """
    Analyze the sentiment of the provided text using VADER.

    Returns:
        Dictionary containing the sentiment label and scores.
    """

    scores = sentiment_analyzer.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label": label,
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "positive": scores["pos"],
        "compound": compound
    }


def predict_text(text):
    """
    Classify text using the trained DistilBERT model and independently
    analyze its sentiment using VADER.

    Args:
        text: News headline, article or social-media post.

    Returns:
        Dictionary containing the predicted category, confidence,
        probability distribution and sentiment analysis.
    """

    text = text.strip()

    if not text:
        raise ValueError("Input text cannot be empty.")

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        output = model(**encoded)

        probabilities_tensor = torch.softmax(
            output.logits,
            dim=1
        )[0]

    predicted_class = int(
        torch.argmax(
            probabilities_tensor
        ).item()
    )

    confidence = float(
        probabilities_tensor[
            predicted_class
        ].item()
    )

    probabilities = (
        probabilities_tensor
        .cpu()
        .tolist()
    )

    probability_by_label = {
        LABELS[i]: float(probabilities[i])
        for i in range(len(probabilities))
    }

    sentiment = analyze_sentiment(text)

    return {
        "predicted_class": predicted_class,
        "predicted_label": LABELS[predicted_class],
        "confidence": confidence,
        "probabilities": probabilities,
        "probability_by_label": probability_by_label,
        "sentiment": sentiment
    }


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("FAKE NEWS CLASSIFICATION SYSTEM")
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
            result = predict_text(text)

            print("\nRESULT")
            print("-" * 50)

            print(
                "Predicted class:",
                result["predicted_class"]
            )

            print(
                "Predicted label:",
                result["predicted_label"]
            )

            print(
                "Confidence:",
                f"{result['confidence'] * 100:.2f}%"
            )

            print("\nClass probabilities:")

            for label, probability in (
                result["probability_by_label"].items()
            ):
                print(
                    f"{label}: "
                    f"{probability * 100:.2f}%"
                )

            print(
                "\nSentiment:",
                result["sentiment"]["label"]
            )

            print(
                "Compound sentiment:",
                result["sentiment"]["compound"]
            )

        except Exception as error:
            print("Error:", error)

        print("\n" + "=" * 70)
