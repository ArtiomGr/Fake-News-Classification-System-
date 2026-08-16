import pandas as pd
import numpy as np
import torch
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from datasets import Dataset
import matplotlib.pyplot as plt


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATA_DIR = Path("data")
MODEL_DIR = Path("models/distilbert")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

NUM_LABELS = 6

# CPU-friendly experiment sizes
TRAIN_SIZE = 12000
VAL_SIZE = 3000
TEST_SIZE = 5000

MAX_LENGTH = 128

print("=" * 70)
print("DISTILBERT FAKE NEWS CLASSIFICATION")
print("=" * 70)

print("\nDevice:", "CUDA" if torch.cuda.is_available() else "CPU")


# -------------------------------------------------
# 1. Load datasets
# -------------------------------------------------

print("\n[1/8] Loading datasets...")

train_df = pd.read_csv(
    DATA_DIR / "all_train.tsv",
    sep="\t",
    usecols=["clean_title", "6_way_label"]
).dropna()

val_df = pd.read_csv(
    DATA_DIR / "all_validate.tsv",
    sep="\t",
    usecols=["clean_title", "6_way_label"]
).dropna()

test_df = pd.read_csv(
    DATA_DIR / "all_test_public.tsv",
    sep="\t",
    usecols=["clean_title", "6_way_label"]
).dropna()


# -------------------------------------------------
# 2. Stratified sampling
# -------------------------------------------------

print("\n[2/8] Creating CPU-friendly stratified samples...")


def stratified_sample(df, size):
    if size >= len(df):
        return df.copy()

    sampled, _ = train_test_split(
        df,
        train_size=size,
        stratify=df["6_way_label"],
        random_state=42
    )

    return sampled.reset_index(drop=True)


train_df = stratified_sample(train_df, TRAIN_SIZE)
val_df = stratified_sample(val_df, VAL_SIZE)
test_df = stratified_sample(test_df, TEST_SIZE)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain label distribution:")
print(train_df["6_way_label"].value_counts().sort_index())


# -------------------------------------------------
# 3. Convert to Hugging Face datasets
# -------------------------------------------------

print("\n[3/8] Preparing Hugging Face datasets...")

train_df = train_df.rename(
    columns={
        "clean_title": "text",
        "6_way_label": "label"
    }
)

val_df = val_df.rename(
    columns={
        "clean_title": "text",
        "6_way_label": "label"
    }
)

test_df = test_df.rename(
    columns={
        "clean_title": "text",
        "6_way_label": "label"
    }
)

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df,
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# -------------------------------------------------
# 4. Tokenizer
# -------------------------------------------------

print("\n[4/8] Loading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")


# -------------------------------------------------
# 5. Model
# -------------------------------------------------

print("\n[5/8] Loading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)


# -------------------------------------------------
# Metrics
# -------------------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# -------------------------------------------------
# 6. Training
# -------------------------------------------------

print("\n[6/8] Training DistilBERT...")
print("This can take a while on CPU.")

training_args = TrainingArguments(
    output_dir="models/distilbert_checkpoints",

    num_train_epochs=2,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,

    learning_rate=2e-5,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    logging_steps=100,

    save_total_limit=1,

    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    compute_metrics=compute_metrics
)

start_time = time.time()

trainer.train()

training_time = time.time() - start_time

print(
    f"\nTraining time: "
    f"{training_time:.2f} seconds"
)


# -------------------------------------------------
# 7. Test evaluation
# -------------------------------------------------

print("\n[7/8] Evaluating test dataset...")

prediction_start = time.time()

output = trainer.predict(test_dataset)

prediction_time = time.time() - prediction_start

predictions = np.argmax(
    output.predictions,
    axis=-1
)

labels = output.label_ids

accuracy = accuracy_score(
    labels,
    predictions
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )
)

print("\nFINAL DISTILBERT RESULTS")
print("-" * 40)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print(
    f"Prediction time: "
    f"{prediction_time:.2f} seconds"
)

print(
    f"Average prediction time: "
    f"{prediction_time / len(test_dataset):.6f} "
    f"seconds/sample"
)

print("\nClassification Report:")

print(
    classification_report(
        labels,
        predictions,
        digits=4,
        zero_division=0
    )
)


# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

cm = confusion_matrix(
    labels,
    predictions
)

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title(
    "DistilBERT Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(range(NUM_LABELS))
plt.yticks(range(NUM_LABELS))

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

plt.savefig(
    RESULTS_DIR /
    "distilbert_confusion_matrix.png",
    dpi=300
)

plt.close()


# -------------------------------------------------
# Save metrics
# -------------------------------------------------

results = pd.DataFrame(
    {
        "Model": ["DistilBERT"],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1": [f1],
        "Training_Time_Seconds": [
            training_time
        ],
        "Prediction_Time_Seconds": [
            prediction_time
        ],
        "Training_Samples": [
            len(train_dataset)
        ],
        "Validation_Samples": [
            len(val_dataset)
        ],
        "Test_Samples": [
            len(test_dataset)
        ]
    }
)

results.to_csv(
    RESULTS_DIR /
    "distilbert_metrics.csv",
    index=False
)


# -------------------------------------------------
# 8. Save final model
# -------------------------------------------------

print("\n[8/8] Saving model...")

trainer.save_model(
    MODEL_DIR
)

tokenizer.save_pretrained(
    MODEL_DIR
)

print("\nSaved:")
print("models/distilbert/")
print("results/distilbert_metrics.csv")
print(
    "results/distilbert_confusion_matrix.png"
)

print("\n" + "=" * 70)

print(
    "DISTILBERT EXPERIMENT "
    "COMPLETED SUCCESSFULLY"
)

print("=" * 70)