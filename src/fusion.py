import pandas as pd
import numpy as np
import torch
import pickle
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import matplotlib.pyplot as plt


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DISTILBERT_DIR = MODEL_DIR / "distilbert"
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_SIZE = 12000
VAL_SIZE = 3000
TEST_SIZE = 5000

MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_LABELS = 6


print("=" * 70)
print("DISTILBERT + SENTIMENT FUSION")
print("=" * 70)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)


# -------------------------------------------------
# 1. Load sentiment datasets
# -------------------------------------------------

print("\n[1/8] Loading sentiment datasets...")

train_df = pd.read_csv(
    DATA_DIR / "train_with_sentiment.csv"
)

val_df = pd.read_csv(
    DATA_DIR / "validate_with_sentiment.csv"
)

test_df = pd.read_csv(
    DATA_DIR / "test_with_sentiment.csv"
)


# -------------------------------------------------
# 2. Recreate same stratified samples
# -------------------------------------------------

print("\n[2/8] Creating stratified samples...")


def stratified_sample(df, size):

    if size >= len(df):
        return df.copy().reset_index(drop=True)

    sampled, _ = train_test_split(
        df,
        train_size=size,
        stratify=df["6_way_label"],
        random_state=42
    )

    return sampled.reset_index(drop=True)


train_df = stratified_sample(
    train_df,
    TRAIN_SIZE
)

val_df = stratified_sample(
    val_df,
    VAL_SIZE
)

test_df = stratified_sample(
    test_df,
    TEST_SIZE
)


print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# -------------------------------------------------
# 3. Load trained DistilBERT
# -------------------------------------------------

print("\n[3/8] Loading trained DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    DISTILBERT_DIR
)

distilbert_model = (
    AutoModelForSequenceClassification
    .from_pretrained(DISTILBERT_DIR)
)

distilbert_model.to(device)

distilbert_model.eval()


# -------------------------------------------------
# 4. Get DistilBERT probabilities
# -------------------------------------------------

print("\n[4/8] Extracting DistilBERT probabilities...")


def get_distilbert_probabilities(texts, name):

    all_probabilities = []

    start_time = time.time()

    texts = texts.tolist()

    for start in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        end = start + BATCH_SIZE

        batch_texts = texts[start:end]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = distilbert_model(
                **encoded
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        if start % 1000 == 0:

            print(
                f"{name}: "
                f"{min(end, len(texts))}"
                f"/{len(texts)}"
            )

    probabilities = np.vstack(
        all_probabilities
    )

    elapsed = time.time() - start_time

    print(
        f"{name} completed in "
        f"{elapsed:.2f} seconds"
    )

    return probabilities, elapsed


train_probs, train_inference_time = (
    get_distilbert_probabilities(
        train_df["clean_title"],
        "Train"
    )
)

val_probs, val_inference_time = (
    get_distilbert_probabilities(
        val_df["clean_title"],
        "Validation"
    )
)

test_probs, test_inference_time = (
    get_distilbert_probabilities(
        test_df["clean_title"],
        "Test"
    )
)


# -------------------------------------------------
# 5. Combine DistilBERT + sentiment
# -------------------------------------------------

print("\n[5/8] Creating fusion features...")


sentiment_columns = [
    "sent_neg",
    "sent_neu",
    "sent_pos",
    "sent_compound"
]


train_sentiment = (
    train_df[sentiment_columns]
    .to_numpy()
)

val_sentiment = (
    val_df[sentiment_columns]
    .to_numpy()
)

test_sentiment = (
    test_df[sentiment_columns]
    .to_numpy()
)


X_train = np.hstack([
    train_probs,
    train_sentiment
])

X_val = np.hstack([
    val_probs,
    val_sentiment
])

X_test = np.hstack([
    test_probs,
    test_sentiment
])


y_train = (
    train_df["6_way_label"]
    .to_numpy()
)

y_val = (
    val_df["6_way_label"]
    .to_numpy()
)

y_test = (
    test_df["6_way_label"]
    .to_numpy()
)


print(
    "Fusion feature size:",
    X_train.shape
)


# -------------------------------------------------
# 6. Train fusion classifier
# -------------------------------------------------

print("\n[6/8] Training fusion classifier...")

fusion_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs"
)

training_start = time.time()

fusion_model.fit(
    X_train,
    y_train
)

fusion_training_time = (
    time.time() - training_start
)


print(
    f"Fusion training time: "
    f"{fusion_training_time:.2f} seconds"
)


# -------------------------------------------------
# Validation
# -------------------------------------------------

val_predictions = fusion_model.predict(
    X_val
)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision, val_recall, val_f1, _ = (
    precision_recall_fscore_support(
        y_val,
        val_predictions,
        average="weighted",
        zero_division=0
    )
)


print("\nVALIDATION RESULTS")
print("-" * 40)

print(
    f"Accuracy : {val_accuracy:.4f}"
)

print(
    f"Precision: {val_precision:.4f}"
)

print(
    f"Recall   : {val_recall:.4f}"
)

print(
    f"F1 Score : {val_f1:.4f}"
)


# -------------------------------------------------
# 7. Test
# -------------------------------------------------

print("\n[7/8] Evaluating fusion model...")

prediction_start = time.time()

test_predictions = fusion_model.predict(
    X_test
)

fusion_prediction_time = (
    time.time() - prediction_start
)


accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_test,
        test_predictions,
        average="weighted",
        zero_division=0
    )
)


print("\nFINAL FUSION RESULTS")
print("-" * 40)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        digits=4,
        zero_division=0
    )
)


# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_predictions
)

plt.figure(
    figsize=(8, 6)
)

plt.imshow(cm)

plt.title(
    "DistilBERT + Sentiment Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    range(NUM_LABELS)
)

plt.yticks(
    range(NUM_LABELS)
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "fusion_confusion_matrix.png",
    dpi=300
)

plt.close()


# -------------------------------------------------
# Save metrics
# -------------------------------------------------

results = pd.DataFrame(
    {
        "Model": [
            "DistilBERT + Sentiment"
        ],

        "Accuracy": [
            accuracy
        ],

        "Precision": [
            precision
        ],

        "Recall": [
            recall
        ],

        "F1": [
            f1
        ],

        "Fusion_Training_Time_Seconds": [
            fusion_training_time
        ],

        "DistilBERT_Test_Inference_Seconds": [
            test_inference_time
        ],

        "Fusion_Prediction_Time_Seconds": [
            fusion_prediction_time
        ],

        "Training_Samples": [
            len(train_df)
        ],

        "Validation_Samples": [
            len(val_df)
        ],

        "Test_Samples": [
            len(test_df)
        ]
    }
)


results.to_csv(
    RESULTS_DIR /
    "fusion_metrics.csv",
    index=False
)


# -------------------------------------------------
# 8. Save fusion model
# -------------------------------------------------

print("\n[8/8] Saving fusion model...")


with open(
    MODEL_DIR /
    "fusion_model.pkl",
    "wb"
) as file:

    pickle.dump(
        fusion_model,
        file
    )


print("\nSaved:")

print(
    "models/fusion_model.pkl"
)

print(
    "results/fusion_metrics.csv"
)

print(
    "results/fusion_confusion_matrix.png"
)


print("\n" + "=" * 70)

print(
    "FUSION EXPERIMENT "
    "COMPLETED SUCCESSFULLY"
)

print("=" * 70)