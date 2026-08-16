import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("SENTIMENT ANALYSIS")
print("=" * 70)

# Load datasets
print("\n[1/4] Loading datasets...")

train_df = pd.read_csv(DATA_DIR / "all_train.tsv", sep="\t")
val_df = pd.read_csv(DATA_DIR / "all_validate.tsv", sep="\t")
test_df = pd.read_csv(DATA_DIR / "all_test_public.tsv", sep="\t")

# Keep only required data
train_df = train_df[["clean_title", "6_way_label"]].dropna().copy()
val_df = val_df[["clean_title", "6_way_label"]].dropna().copy()
test_df = test_df[["clean_title", "6_way_label"]].dropna().copy()

analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    scores = analyzer.polarity_scores(str(text))

    return pd.Series({
        "sent_neg": scores["neg"],
        "sent_neu": scores["neu"],
        "sent_pos": scores["pos"],
        "sent_compound": scores["compound"]
    })


def process_dataset(df, name):
    print(f"\nProcessing {name}...")

    start = time.time()

    sentiment_features = df["clean_title"].apply(get_sentiment)

    result = pd.concat(
        [df.reset_index(drop=True),
         sentiment_features.reset_index(drop=True)],
        axis=1
    )

    elapsed = time.time() - start

    print(f"{name}: {len(result)} samples")
    print(f"Completed in {elapsed:.2f} seconds")

    return result


print("\n[2/4] Extracting sentiment features...")

train_sent = process_dataset(train_df, "Train")
val_sent = process_dataset(val_df, "Validation")
test_sent = process_dataset(test_df, "Test")


print("\n[3/4] Saving sentiment datasets...")

train_sent.to_csv(
    DATA_DIR / "train_with_sentiment.csv",
    index=False
)

val_sent.to_csv(
    DATA_DIR / "validate_with_sentiment.csv",
    index=False
)

test_sent.to_csv(
    DATA_DIR / "test_with_sentiment.csv",
    index=False
)


print("\n[4/4] Sentiment summary...")

print("\nAverage sentiment by fake-news label:")

summary = train_sent.groupby("6_way_label")[
    ["sent_neg", "sent_neu", "sent_pos", "sent_compound"]
].mean()

print(summary)

summary.to_csv(
    RESULTS_DIR / "sentiment_by_label.csv"
)


print("\nSaved:")
print("data/train_with_sentiment.csv")
print("data/validate_with_sentiment.csv")
print("data/test_with_sentiment.csv")
print("results/sentiment_by_label.csv")

print("\n" + "=" * 70)
print("SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)