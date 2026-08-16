import torch
import numpy as np
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


MODEL_DIR = Path("models/distilbert")

LABELS = {
    0: "True",
    1: "Satire",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content"
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(device)
model.eval()

sentiment_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):

    scores = sentiment_analyzer.polarity_scores(
        text
    )

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

        output = model(
            **encoded
        )

        probabilities = torch.softmax(
            output.logits,
            dim=1
        )[0]

    predicted_class = int(
        torch.argmax(
            probabilities
        ).item()
    )

    confidence = float(
        probabilities[predicted_class]
        .item()
    )

    probabilities = (
        probabilities
        .cpu()
        .numpy()
    )

    sentiment = analyze_sentiment(
        text
    )

    return {
        "predicted_class": predicted_class,
        "predicted_label": LABELS[predicted_class],
        "confidence": confidence,
        "probabilities": probabilities,
        "sentiment": sentiment
    }


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

            print(
                "Please enter valid text."
            )

            continue

        result = predict_text(
            text
        )

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

        print(
            "\nClass probabilities:"
        )

        for i, probability in enumerate(
            result["probabilities"]
        ):

            print(
                f"Class {i}: "
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

        print("\n" + "=" * 70)