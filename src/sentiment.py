import time
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

_analyzer = SentimentIntensityAnalyzer()


def analyze_vader_sentiment(text):
    """Analyze emotional tone for a single text using VADER.

    Returns a dictionary with sentiment label and component scores.
    The sentiment result is intentionally kept separate from the three
    fake-news classifiers.
    """
    clean_text = " ".join(str(text).split()) if text is not None else ""

    if not clean_text:
        return {
            "sentiment_label": "Neutral",
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
            "compound": 0.0,
        }

    scores = _analyzer.polarity_scores(clean_text)
    compound = float(scores.get("compound", 0.0))
    positive = float(scores.get("pos", 0.0))
    neutral = float(scores.get("neu", 0.0))
    negative = float(scores.get("neg", 0.0))

    if compound >= 0.05:
        sentiment_label = "Positive"
    elif compound <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return {
        "sentiment_label": sentiment_label,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "compound": compound,
    }


def get_sentiment(text):
    """Backward-compatible wrapper for the older dataset-prep pipeline."""
    scores = analyze_vader_sentiment(text)
    return pd.Series({
        "sent_neg": scores["negative"],
        "sent_neu": scores["neutral"],
        "sent_pos": scores["positive"],
        "sent_compound": scores["compound"],
    })


def process_dataset(df, name):
    print(f"\nProcessing {name}...")

    start = time.time()
    sentiment_features = df["clean_title"].apply(get_sentiment)

    result = pd.concat(
        [df.reset_index(drop=True), sentiment_features.reset_index(drop=True)],
        axis=1,
    )

    elapsed = time.time() - start
    print(f"{name}: {len(result)} samples")
    print(f"Completed in {elapsed:.2f} seconds")
    return result


def main():
    print("=" * 70)
    print("SENTIMENT ANALYSIS")
    print("=" * 70)

    print("\n[1/4] Loading datasets...")
    train_df = pd.read_csv(DATA_DIR / "all_train.tsv", sep="\t")
    val_df = pd.read_csv(DATA_DIR / "all_validate.tsv", sep="\t")
    test_df = pd.read_csv(DATA_DIR / "all_test_public.tsv", sep="\t")

    train_df = train_df[["clean_title", "6_way_label"]].dropna().copy()
    val_df = val_df[["clean_title", "6_way_label"]].dropna().copy()
    test_df = test_df[["clean_title", "6_way_label"]].dropna().copy()

    print("\n[2/4] Extracting sentiment features...")
    train_sent = process_dataset(train_df, "Train")
    val_sent = process_dataset(val_df, "Validation")
    test_sent = process_dataset(test_df, "Test")

    print("\n[3/4] Saving sentiment datasets...")
    train_sent.to_csv(DATA_DIR / "train_with_sentiment.csv", index=False)
    val_sent.to_csv(DATA_DIR / "validate_with_sentiment.csv", index=False)
    test_sent.to_csv(DATA_DIR / "test_with_sentiment.csv", index=False)

    print("\n[4/4] Sentiment summary...")
    print("\nAverage sentiment by fake-news label:")
    summary = train_sent.groupby("6_way_label")[
        ["sent_neg", "sent_neu", "sent_pos", "sent_compound"]
    ].mean()
    print(summary)
    summary.to_csv(RESULTS_DIR / "sentiment_by_label.csv")

    print("\nSaved:")
    print("data/train_with_sentiment.csv")
    print("data/validate_with_sentiment.csv")
    print("data/test_with_sentiment.csv")
    print("results/sentiment_by_label.csv")

    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()