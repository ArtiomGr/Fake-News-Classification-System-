import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import pickle
import time


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


print("=" * 70)
print("BASELINE MODEL")
print("TF-IDF + LOGISTIC REGRESSION")
print("=" * 70)


# -------------------------------------------------
# 1. Load data
# -------------------------------------------------

print("\n[1/7] Loading dataset...")

train_df = pd.read_csv(DATA_DIR / "all_train.tsv", sep="\t")
val_df = pd.read_csv(DATA_DIR / "all_validate.tsv", sep="\t")
test_df = pd.read_csv(DATA_DIR / "all_test_public.tsv", sep="\t")


# -------------------------------------------------
# 2. Keep required columns
# -------------------------------------------------

print("[2/7] Preparing data...")

train_df = train_df[["clean_title", "6_way_label"]].copy()
val_df = val_df[["clean_title", "6_way_label"]].copy()
test_df = test_df[["clean_title", "6_way_label"]].copy()

train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()

train_df["clean_title"] = train_df["clean_title"].astype(str)
val_df["clean_title"] = val_df["clean_title"].astype(str)
test_df["clean_title"] = test_df["clean_title"].astype(str)


X_train = train_df["clean_title"]
y_train = train_df["6_way_label"]

X_val = val_df["clean_title"]
y_val = val_df["6_way_label"]

X_test = test_df["clean_title"]
y_test = test_df["6_way_label"]


print("Train samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))


# -------------------------------------------------
# 3. TF-IDF
# -------------------------------------------------

print("\n[3/7] Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words="english",
    sublinear_tf=True
)

start_time = time.time()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF feature shape:", X_train_tfidf.shape)


# -------------------------------------------------
# 4. Train Logistic Regression
# -------------------------------------------------

print("\n[4/7] Training Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs"
)

model.fit(X_train_tfidf, y_train)

training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds")


# -------------------------------------------------
# 5. Validation
# -------------------------------------------------

print("\n[5/7] Validation results...")

val_predictions = model.predict(X_val_tfidf)

val_accuracy = accuracy_score(y_val, val_predictions)

val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
    y_val,
    val_predictions,
    average="weighted",
    zero_division=0
)

print(f"Validation Accuracy : {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall   : {val_recall:.4f}")
print(f"Validation F1       : {val_f1:.4f}")


# -------------------------------------------------
# 6. Test
# -------------------------------------------------

print("\n[6/7] Test results...")

prediction_start = time.time()

test_predictions = model.predict(X_test_tfidf)

prediction_time = time.time() - prediction_start

accuracy = accuracy_score(y_test, test_predictions)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    test_predictions,
    average="weighted",
    zero_division=0
)

print("\nFINAL TEST RESULTS")
print("-" * 40)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print(f"\nPrediction time: {prediction_time:.4f} seconds")
print(
    f"Average prediction time: "
    f"{prediction_time / len(X_test):.6f} seconds/sample"
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

cm = confusion_matrix(y_test, test_predictions)

plt.figure(figsize=(8, 6))
plt.imshow(cm)

plt.title("Baseline Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(range(6))
plt.yticks(range(6))

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

confusion_path = RESULTS_DIR / "baseline_confusion_matrix.png"

plt.savefig(confusion_path, dpi=300)
plt.close()


# -------------------------------------------------
# Save results
# -------------------------------------------------

results = pd.DataFrame(
    {
        "Model": ["TF-IDF + Logistic Regression"],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1": [f1],
        "Training_Time_Seconds": [training_time],
        "Prediction_Time_Seconds": [prediction_time]
    }
)

results.to_csv(
    RESULTS_DIR / "baseline_metrics.csv",
    index=False
)


# -------------------------------------------------
# 7. Save model
# -------------------------------------------------

print("\n[7/7] Saving model...")

with open(MODEL_DIR / "baseline_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("\nSaved:")
print("models/baseline_model.pkl")
print("models/tfidf_vectorizer.pkl")
print("results/baseline_metrics.csv")
print("results/baseline_confusion_matrix.png")

print("\n" + "=" * 70)
print("BASELINE EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)