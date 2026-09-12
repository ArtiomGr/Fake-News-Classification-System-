import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# -------------------------------------------------
# Load metrics
# -------------------------------------------------

baseline = pd.read_csv(
    RESULTS_DIR / "baseline_metrics.csv"
)

distilbert = pd.read_csv(
    RESULTS_DIR / "distilbert_metrics.csv"
)

bert = pd.read_csv(
    RESULTS_DIR / "BERT_metrics.csv"
)

fusion = pd.read_csv(
    RESULTS_DIR / "fusion_metrics.csv"
)


# -------------------------------------------------
# Keep common metrics
# -------------------------------------------------

baseline_row = {
    "Model": "TF-IDF + Logistic Regression",
    "Accuracy": baseline.loc[0, "Accuracy"],
    "Precision": baseline.loc[0, "Precision"],
    "Recall": baseline.loc[0, "Recall"],
    "F1": baseline.loc[0, "F1"]
}

distilbert_row = {
    "Model": "DistilBERT",
    "Accuracy": distilbert.loc[0, "Accuracy"],
    "Precision": distilbert.loc[0, "Precision"],
    "Recall": distilbert.loc[0, "Recall"],
    "F1": distilbert.loc[0, "F1"]
}

bert_row = {
    "Model": "BERT",
    "Accuracy": bert.loc[0, "Accuracy"],
    "Precision": bert.loc[0, "Precision"],
    "Recall": bert.loc[0, "Recall"],
    "F1": bert.loc[0, "F1"]
}

fusion_row = {
    "Model": "DistilBERT + Sentiment",
    "Accuracy": fusion.loc[0, "Accuracy"],
    "Precision": fusion.loc[0, "Precision"],
    "Recall": fusion.loc[0, "Recall"],
    "F1": fusion.loc[0, "F1"]
}


# -------------------------------------------------
# Create comparison table
# -------------------------------------------------

comparison = pd.DataFrame([
    baseline_row,
    distilbert_row,
    bert_row,
    fusion_row
])

print("\nFINAL MODEL COMPARISON")
print("-" * 70)

print(
    comparison.to_string(
        index=False
    )
)

comparison.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)


# -------------------------------------------------
# Accuracy graph
# -------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title(
    "Model Accuracy Comparison"
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

for i, value in enumerate(
    comparison["Accuracy"]
):
    plt.text(
        i,
        value + 0.01,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "accuracy_comparison.png",
    dpi=300
)

plt.close()


# -------------------------------------------------
# F1 graph
# -------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["F1"]
)

plt.title(
    "Model F1 Score Comparison"
)

plt.ylabel(
    "Weighted F1 Score"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

for i, value in enumerate(
    comparison["F1"]
):
    plt.text(
        i,
        value + 0.01,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "f1_comparison.png",
    dpi=300
)

plt.close()


# -------------------------------------------------
# All metrics graph
# -------------------------------------------------

comparison_plot = comparison.set_index(
    "Model"
)[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
]

comparison_plot.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title(
    "Overall Model Performance"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR /
    "overall_model_comparison.png",
    dpi=300
)

plt.close()


# -------------------------------------------------
# Best model
# -------------------------------------------------

best_accuracy_row = comparison.loc[
    comparison["Accuracy"].idxmax()
]

best_f1_row = comparison.loc[
    comparison["F1"].idxmax()
]

print("\nBest Accuracy:")
print(
    best_accuracy_row["Model"],
    f"({best_accuracy_row['Accuracy']:.4f})"
)

print("\nBest F1:")
print(
    best_f1_row["Model"],
    f"({best_f1_row['F1']:.4f})"
)


# -------------------------------------------------
# Save summary
# -------------------------------------------------

summary = pd.DataFrame(
    {
        "Metric": [
            "Best Accuracy Model",
            "Best Accuracy",
            "Best F1 Model",
            "Best F1"
        ],
        "Value": [
            best_accuracy_row["Model"],
            best_accuracy_row["Accuracy"],
            best_f1_row["Model"],
            best_f1_row["F1"]
        ]
    }
)

summary.to_csv(
    RESULTS_DIR / "best_model_summary.csv",
    index=False
)


print("\nSaved:")
print("results/model_comparison.csv")
print("results/accuracy_comparison.png")
print("results/f1_comparison.png")
print("results/overall_model_comparison.png")
print("results/best_model_summary.csv")

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 70)