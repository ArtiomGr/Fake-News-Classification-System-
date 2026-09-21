"""Shared paths, immutable experiment inputs and final-task evaluation helpers."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/final_six_category_dataset.csv"
DISTIL_RESULTS = ROOT / "results/distilbert_six_category_native_ads_corrected"
SPLIT_FILE = DISTIL_RESULTS / "split_manifest.csv"
MODEL_PATHS = {
    "BERT": ROOT / "models/bert_six_category_final",
    "DistilBERT": ROOT / "models/distilbert_six_category_native_ads_corrected",
    "Logistic Regression": ROOT / "models/logistic_regression_six_category_final",
}
RESULT_PATHS = {name: ROOT / "results" / path.name for name, path in MODEL_PATHS.items()}
MAPPING = json.loads((MODEL_PATHS["DistilBERT"] / "label_mapping.json").read_text(encoding="utf-8"))
CATEGORIES = {int(key): value for key, value in MAPPING["project_id_to_category"].items()}
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def frozen_data():
    """Read existing final rows/partitions; never clean, resample or regenerate."""
    plan = read_json(DISTIL_RESULTS / "training_plan.json")
    if sha256(DATASET) != plan["dataset_sha256"] or sha256(SPLIT_FILE) != plan["split_manifest_sha256"]:
        raise ValueError("Final dataset or frozen manifest changed")
    with DATASET.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with SPLIT_FILE.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(rows) != 8459 or len(manifest) != len(rows):
        raise ValueError("Final row counts do not match")
    groups = {}
    indices = {key: [] for key in ("train", "validation", "test")}
    for index, (row, entry) in enumerate(zip(rows, manifest)):
        if (int(entry["dataset_row"]) != index or entry["text_group_id"] != row["text_group_id"]
                or entry["project_label"] != row["final_label"]
                or int(entry["internal_label"]) != int(row["final_label"]) - 1
                or CATEGORIES[int(row["final_label"])] != row["final_category"]):
            raise ValueError("Frozen row identity/label mismatch")
        split = entry["split"]
        if groups.setdefault(entry["group_id"], split) != split:
            raise ValueError("Related group crosses frozen partitions")
        indices[split].append(index)
    if {key: len(value) for key, value in indices.items()} != plan["split_totals"]:
        raise ValueError("Partition sizes differ from the completed experiment")
    return rows, indices, plan


def evaluate(split, probabilities, rows, indices, destination):
    """Save the same five metrics, per-class report, matrix and row predictions."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    labels = np.array([int(rows[i]["final_label"]) - 1 for i in indices])
    predictions = np.asarray(probabilities).argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)[2]
    metrics = dict(zip(METRICS, map(float, [accuracy_score(labels, predictions), precision, recall, f1, weighted])))
    metrics["records"] = len(indices)
    write_json(destination / f"{split}_metrics.json", metrics)
    names = list(CATEGORIES.values())
    report = classification_report(labels, predictions, labels=list(range(6)), target_names=names,
                                   output_dict=True, zero_division=0)
    write_json(destination / f"{split}_classification_report.json", report)
    with (destination / f"{split}_per_class_metrics.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["project_label", "category", "precision", "recall", "f1", "support"])
        for project_id, name in CATEGORIES.items():
            r = report[name]
            writer.writerow([project_id, name, r["precision"], r["recall"], r["f1-score"], r["support"]])
    matrix = confusion_matrix(labels, predictions, labels=list(range(6)))
    with (destination / f"{split}_confusion_matrix.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_category / predicted_category", *names])
        writer.writerows([[name, *row] for name, row in zip(names, matrix)])
    with (destination / f"{split}_predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_row", "text_group_id", "true_project_label", "true_category", "predicted_project_label", "predicted_category"])
        writer.writerows([[i, rows[i]["text_group_id"], int(y) + 1, CATEGORIES[int(y) + 1],
                           int(p) + 1, CATEGORIES[int(p) + 1]] for i, y, p in zip(indices, labels, predictions)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(matrix, display_labels=names).plot(ax=ax, xticks_rotation=35, colorbar=False)
    ax.set_title(f"{destination.name}: {split}")
    fig.tight_layout()
    fig.savefig(destination / f"{split}_confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(split, json.dumps(metrics), flush=True)
    return metrics
