"""Prepare or train the new six-category DistilBERT experiment, entirely offline.

Default/--prepare: validate the immutable CSV and save grouped stratified split
indices, label mappings, and the training plan. No model is loaded or trained.
--train: use that frozen split and the locally cached original pretrained base.
Adapted from train_distilbert.py's tokenizer/Dataset/Trainer/metrics workflow;
the old script cannot be imported safely because it trains at module scope.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import unicodedata
from collections import Counter, defaultdict
from urllib.parse import urlsplit

# Disallow Hub/telemetry requests, including during preparation and later training.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/final_six_category_dataset.csv"
MODEL_DIR = ROOT / "models/distilbert_six_category"
RESULTS_DIR = ROOT / "results/distilbert_six_category"
SPLIT_FILE = RESULTS_DIR / "split_manifest.csv"
PLAN_FILE = RESULTS_DIR / "training_plan.json"
MAPPING_FILE = RESULTS_DIR / "label_mapping.json"
BASE_MODEL = "distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
SEED = 42
SPLITS = ("train", "validation", "test")
RATIOS = np.array([0.70, 0.15, 0.15])
CATEGORIES = {1: "Native Advertising", 2: "News Satire", 3: "Propaganda",
              4: "Manipulation", 5: "News Parody", 6: "Fabrication"}
HYPERPARAMETERS = {
    "max_length": 512, "epochs": 3, "learning_rate": 2e-5,
    "weight_decay": 0.01, "train_batch_size": 8, "eval_batch_size": 16,
    "gradient_accumulation_steps": 2, "warmup_fraction": 0.10,
    "lr_scheduler_type": "linear", "optimizer": "adamw_torch",
    "best_model_metric": "macro_f1", "seed": SEED,
    "dynamic_padding": True, "full_determinism": True,
    "fp16": False, "bf16": False, "dataloader_num_workers": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def lexical_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", normalized_text(text)))


def propaganda_template(text: str) -> str:
    """Group superficial prefix/suffix variants without editing model input."""
    text = lexical_text(text)
    text = re.sub(r"^(?:fyi|alert|report|update|breaking)\s+", "", text)
    text = re.sub(r"\s+(?:details inside|full story|breaking news|unconfirmed|"
                  r"report says|witness report|say sources|experts claim)$", "", text)
    return text


def base_snapshot() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE
    snapshot = Path(HF_HUB_CACHE) / f"models--{BASE_MODEL}" / "snapshots" / BASE_REVISION
    for name in ["config.json", "model.safetensors", "tokenizer.json"]:
        if not (snapshot / name).is_file():
            raise FileNotFoundError(f"Required offline pretrained asset missing: {snapshot / name}")
    return snapshot


def load_rows() -> list[dict]:
    with DATASET.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {"text", "final_label", "final_category", "source", "original_id", "text_group_id"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError("Dataset lacks required columns")
        rows = list(reader)
    seen = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV record")
        label = int(row["final_label"])
        if CATEGORIES.get(label) != row["final_category"]:
            raise ValueError("Invalid label/category mapping")
        key = normalized_text(row["text"])
        if not key or key in seen:
            raise ValueError("Empty/duplicate normalized text: dataset is not changed automatically")
        seen.add(key)
    if {int(row["final_label"]) for row in rows} != set(CATEGORIES):
        raise ValueError("Expected all six categories")
    return rows


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while item != self.parent[item]:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> bool:
        left, right = self.find(left), self.find(right)
        if left == right:
            return False
        self.parent[max(left, right)] = min(left, right)
        return True


def make_groups(rows: list[dict]) -> tuple[list[str], dict]:
    """Connect text equivalents, provenance links, queries, and template families.

    Author/subreddit/domain alone do not imply the same underlying record and
    are not grouping keys. These fields are also never passed to the model.
    """
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(base_snapshot() / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    encodings = tokenizer.encode_batch([row["text"] for row in rows])
    lengths = [len(encoded.ids) for encoded in encodings]
    if max(lengths) > HYPERPARAMETERS["max_length"]:
        raise ValueError("Some full texts exceed max_length; review truncation before preparing")
    wanted = {row["original_id"] for row in rows if row["source"] == "manipulation"}
    openings = {}
    raw_path = ROOT / "data/raw/Manipulation/manipulational_conversation.jsonl"
    with raw_path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw["conversation_id"] not in wanted:
                continue
            statements = [m["text"] for m in raw["messages"] if m["speaker"] == "A"]
            if not statements:
                raise ValueError("Manipulation record lacks an A-speaker statement")
            opening = lexical_text(statements[0])
            opening = re.sub(r"^(?:(?:um|well|so|i mean)\s+)+", "", opening)
            openings[raw["conversation_id"]] = opening
    if openings.keys() != wanted:
        raise ValueError("Cannot recover manipulation opening-template metadata")

    union = UnionFind(len(rows))
    owners = {}
    links = Counter()
    for index, (row, encoding) in enumerate(zip(rows, encodings)):
        keys = [("normalized", normalized_text(row["text"])),
                ("lexical", lexical_text(row["text"])),
                ("token_ids", tuple(encoding.ids)),
                ("dataset_group", row["text_group_id"])]
        ids = set(json.loads(row.get("matching_original_ids") or "[]"))
        ids.add(row["original_id"])
        ids.add(row.get("linked_submission_id", ""))
        for identifier in ids - {""}:
            keys.append(("source_record", row["source"] + ":" + identifier.removeprefix("t3_")))
        if row["source"] == "native_ads" and row.get("query"):
            keys.append(("native_query", lexical_text(row["query"])))
        if row["source"] == "propaganda":
            keys.append(("propaganda_template", propaganda_template(row["text"])))
        if row["source"] == "manipulation":
            keys.append(("manipulation_opening", openings[row["original_id"]]))
        if row.get("image_url"):
            url = urlsplit(row["image_url"])
            keys.append(("image_asset", url.netloc.lower() + url.path))
        for kind, value in keys:
            key = (kind, value)
            if key in owners:
                if union.union(index, owners[key]):
                    links[kind] += 1
            else:
                owners[key] = index
    members = defaultdict(list)
    for index in range(len(rows)):
        members[union.find(index)].append(index)
    group_ids = [""] * len(rows)
    for group in members.values():
        identity = "\n".join(sorted(rows[i]["text_group_id"] for i in group))
        group_id = hashlib.sha256(identity.encode()).hexdigest()
        for index in group:
            group_ids[index] = group_id
    audit = {
        "groups": len(members), "multi_record_groups": sum(len(g) > 1 for g in members.values()),
        "largest_group": max(map(len, members.values())), "union_links_by_rule": dict(sorted(links.items())),
        "manipulation_grouping_input_sha256": sha256_file(raw_path),
        "max_full_text_tokens": max(lengths), "truncated_records": 0,
        "tokens_by_source": {},
    }
    for source in sorted({row["source"] for row in rows}):
        values = sorted(n for row, n in zip(rows, lengths) if row["source"] == source)
        audit["tokens_by_source"][source] = {
            "records": len(values), "median": values[len(values) // 2],
            "p95": values[int(len(values) * .95)], "max": max(values),
        }
    return group_ids, audit


def split_targets(labels: np.ndarray) -> np.ndarray:
    """Largest-remainder rounding yields exact per-class integer targets."""
    targets = np.zeros((6, 3), dtype=int)
    for label in range(6):
        ideal = int((labels == label).sum()) * RATIOS
        counts = np.floor(ideal).astype(int)
        order = np.argsort(-(ideal - counts), kind="stable")
        counts[order[:int(round(ideal.sum())) - counts.sum()]] += 1
        targets[label] = counts
    return targets


def grouped_stratified_split(labels: np.ndarray, group_ids: list[str]) -> list[str]:
    """Solve exact class quotas while treating each connected group as atomic.

    Only multi-record groups need binary assignment variables. Singletons are
    represented as integer counts per class/split, then shuffled with seed 42.
    No group is broken or row discarded if targets are infeasible: fail clearly.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    groups = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        groups[group_id].append(index)
    multi = [indices for _, indices in sorted(groups.items()) if len(indices) > 1]
    single = defaultdict(list)
    for _, indices in sorted(groups.items()):
        if len(indices) == 1:
            single[int(labels[indices[0]])].append(indices[0])
    count = len(multi)
    variables = count * 3 + 18
    matrix = lil_matrix((count + 18 + 6, variables), dtype=float)
    rhs = np.zeros(matrix.shape[0])
    upper = np.ones(variables)
    targets = split_targets(labels)
    for group_index, indices in enumerate(multi):
        matrix[group_index, group_index * 3:group_index * 3 + 3] = 1
        rhs[group_index] = 1
        class_counts = np.bincount(labels[indices], minlength=6)
        for label in range(6):
            for split in range(3):
                matrix[count + label * 3 + split, group_index * 3 + split] = class_counts[label]
    for label in range(6):
        for split in range(3):
            variable = count * 3 + label * 3 + split
            matrix[count + label * 3 + split, variable] = 1
            rhs[count + label * 3 + split] = targets[label, split]
            matrix[count + 18 + label, variable] = 1
            upper[variable] = len(single[label])
        rhs[count + 18 + label] = len(single[label])
    rng = np.random.default_rng(SEED)
    cost = np.zeros(variables)
    cost[:count * 3] = rng.random(count * 3)
    solution = milp(cost, integrality=np.ones(variables), bounds=Bounds(np.zeros(variables), upper),
                    constraints=LinearConstraint(matrix.tocsc(), rhs, rhs),
                    options={"time_limit": 60, "mip_rel_gap": 0.0})
    if solution.x is None:
        raise ValueError(f"Grouped 70/15/15 split infeasible or solver timed out: {solution.message}")
    assignment = np.rint(solution.x).astype(int)
    if not np.allclose(matrix @ assignment, rhs):
        raise ValueError("Solver did not produce a valid integral split")
    output = [""] * len(labels)
    for group_index, indices in enumerate(multi):
        split = int(np.argmax(assignment[group_index * 3:group_index * 3 + 3]))
        for index in indices:
            output[index] = SPLITS[split]
    for label in range(6):
        rng.shuffle(single[label])
        start = 0
        for split in range(3):
            size = assignment[count * 3 + label * 3 + split]
            for index in single[label][start:start + size]:
                output[index] = SPLITS[split]
            start += size
    return output


def verify_split(rows, group_ids, splits) -> dict:
    if len(rows) != len(group_ids) or len(rows) != len(splits):
        raise ValueError("Split length mismatch")
    group_owner, text_owner = {}, {}
    actual = np.zeros((6, 3), dtype=int)
    for row, group, split in zip(rows, group_ids, splits):
        if split not in SPLITS:
            raise ValueError("Unknown split")
        for key, owners in [(group, group_owner), (normalized_text(row["text"]), text_owner)]:
            if key in owners and owners[key] != split:
                raise ValueError("Related group/text crosses splits")
            owners[key] = split
        actual[int(row["final_label"]) - 1, SPLITS.index(split)] += 1
    labels = np.array([int(row["final_label"]) - 1 for row in rows])
    if not np.array_equal(actual, split_targets(labels)):
        raise ValueError("Per-class 70/15/15 counts do not match targets")
    return {CATEGORIES[label + 1]: dict(zip(SPLITS, map(int, actual[label]))) for label in range(6)}


def training_class_weights(labels) -> list[float]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=6)
    if len(counts) != 6 or np.any(counts == 0):
        raise ValueError("Training partition must contain all six classes")
    return (counts.sum() / (6 * counts)).tolist()


def label_mapping() -> dict:
    return {"project_id_to_category": {str(k): v for k, v in CATEGORIES.items()},
            "internal_id_to_category": {str(k - 1): v for k, v in CATEGORIES.items()},
            "category_to_internal_id": {v: k - 1 for k, v in CATEGORIES.items()},
            "internal_id_to_project_id": {str(k - 1): k for k in CATEGORIES}}


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare(reuse_split_manifest: Path | None = None) -> tuple[list[dict], list[str], dict]:
    before = sha256_file(DATASET)
    rows = load_rows()
    groups, grouping_audit = make_groups(rows)
    labels = np.array([int(row["final_label"]) - 1 for row in rows])
    if PLAN_FILE.exists():
        plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
        if plan["dataset_sha256"] != before or plan["hyperparameters"] != HYPERPARAMETERS:
            raise ValueError("Dataset/config differs from the frozen preparation plan")
        if plan["grouping"] != grouping_audit:
            raise ValueError("Grouping metadata differs from the frozen preparation plan")
        if plan["split_manifest_sha256"] != sha256_file(SPLIT_FILE):
            raise ValueError("Frozen split manifest has changed")
        with SPLIT_FILE.open(encoding="utf-8", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        if len(manifest) != len(rows):
            raise ValueError("Manifest row count mismatch")
        for index, (item, row, group) in enumerate(zip(manifest, rows, groups)):
            if (int(item["dataset_row"]) != index or item["group_id"] != group
                    or item["text_group_id"] != row["text_group_id"]
                    or item["project_label"] != row["final_label"]):
                raise ValueError("Frozen manifest does not match this dataset/grouping")
        if json.loads(MAPPING_FILE.read_text(encoding="utf-8")) != label_mapping():
            raise ValueError("Saved label mapping differs from the project categories")
        splits = [item["split"] for item in manifest]
        verify_split(rows, groups, splits)
    else:
        if any(path.exists() for path in [SPLIT_FILE, MAPPING_FILE]):
            raise FileExistsError("Incomplete preparation artifacts already exist")
        if reuse_split_manifest is None:
            splits = grouped_stratified_split(labels, groups)
        else:
            with reuse_split_manifest.open(encoding="utf-8", newline="") as handle:
                previous = list(csv.DictReader(handle))
            if len(previous) != len(rows) or any(
                    int(item["dataset_row"]) != index or item["project_label"] != row["final_label"]
                    for index, (item, row) in enumerate(zip(previous, rows))):
                raise ValueError("Previous split manifest does not match row positions/labels")
            splits = [item["split"] for item in previous]
        counts = verify_split(rows, groups, splits)
        train_labels = labels[np.array(splits) == "train"]
        weights = training_class_weights(train_labels)
        steps_per_epoch = math.ceil(math.ceil(len(train_labels) / HYPERPARAMETERS["train_batch_size"])
                                    / HYPERPARAMETERS["gradient_accumulation_steps"])
        import torch
        plan = {
            "dataset": str(DATASET), "dataset_sha256": before, "records": len(rows),
            "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "base_snapshot": str(base_snapshot()), "model_output": str(MODEL_DIR),
            "results_output": str(RESULTS_DIR), "random_state": SEED,
            "hyperparameters": HYPERPARAMETERS, "split_counts": counts,
            "split_totals": dict(Counter(splits)), "grouping": grouping_audit,
            "training_only_class_weights": dict(zip(CATEGORIES.values(), weights)),
            "warmup_steps": math.ceil(steps_per_epoch * HYPERPARAMETERS["epochs"] * HYPERPARAMETERS["warmup_fraction"]),
            "device_available": "cuda" if torch.cuda.is_available() else "cpu",
            "versions": {name: importlib.metadata.version(name) for name in
                         ["torch", "transformers", "datasets", "accelerate", "scikit-learn", "scipy", "numpy"]},
            "leakage_checks": {"normalized_text_overlap": 0, "related_group_overlap": 0,
                               "model_inputs": ["input_ids", "attention_mask"],
                               "model_initialization": "Original pretrained base, fresh classifier; NOT old fine-tuned model"},
            "limitations": [
                "Dataset labels are heuristic, not independently human-verified.",
                "Grouping addresses known templates/IDs/queries/assets; arbitrary semantic paraphrases may remain.",
                "Manipulation groups share an opening template; other phrases may still recur across groups.",
                "Source/style artifacts can predict labels even though source metadata is never an input.",
                "Original source splits are provenance only; these are new grouped train/validation/test partitions.",
                "Authors, domains and subreddits are not held out wholesale; evaluation is not unseen-source generalization.",
            ],
        }
        if reuse_split_manifest is not None:
            plan["preserved_split_manifest"] = str(reuse_split_manifest)
            plan["preserved_split_manifest_sha256"] = sha256_file(reuse_split_manifest)
            plan["limitations"].append(
                "Previous test/sanity results informed Native-Ads correction; this is regression evaluation, not an untouched test.")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with SPLIT_FILE.open("x", encoding="utf-8", newline="") as handle:
            fields = ["dataset_row", "split", "group_id", "text_group_id", "project_label", "internal_label"]
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for index, (row, group, split) in enumerate(zip(rows, groups, splits)):
                writer.writerow(dict(zip(fields, [index, split, group, row["text_group_id"],
                                                  row["final_label"], int(row["final_label"]) - 1])))
        plan["split_manifest_sha256"] = sha256_file(SPLIT_FILE)
        write_json(MAPPING_FILE, label_mapping())
        write_json(PLAN_FILE, plan)
    if sha256_file(DATASET) != before:
        raise RuntimeError("Dataset changed during preparation")
    print(json.dumps({key: plan[key] for key in ["split_counts", "split_totals", "training_only_class_weights",
                                               "grouping", "device_available"]}, indent=2), flush=True)
    return rows, splits, plan


def metric_values(labels, predictions) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=list(range(6)), average="macro", zero_division=0)
    weighted_f1 = precision_recall_fscore_support(
        labels, predictions, labels=list(range(6)), average="weighted", zero_division=0)[2]
    return {"accuracy": float(accuracy_score(labels, predictions)), "macro_precision": float(precision),
            "macro_recall": float(recall), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1)}


def classification_loss(logits, labels, class_weights, training: bool, num_items_in_batch=None):
    """Train-only weighted CE, normalized by examples across accumulated batches.

    Balanced weights N/(6*n_class) average to one over the training partition.
    Evaluation loss is ordinary unweighted cross entropy; all reported metrics
    operate on untouched validation/test samples without reweighting.
    """
    import torch
    from torch.nn.functional import cross_entropy
    weights = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device) if training else None
    loss = cross_entropy(logits, labels, weight=weights, reduction="sum")
    denominator = num_items_in_batch if training and num_items_in_batch is not None else labels.numel()
    return loss / denominator


def model_dataset(rows, indices, tokenizer):
    """Allow only tokenized text and the supervised target into Trainer."""
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": [rows[i]["text"] for i in indices],
                                 "labels": [int(rows[i]["final_label"]) - 1 for i in indices]})
    dataset = dataset.map(lambda batch: tokenizer(batch["text"], truncation=True,
                          max_length=HYPERPARAMETERS["max_length"], padding=False,
                          return_token_type_ids=False),
                          batched=True, remove_columns=["text"], load_from_cache_file=False)
    if set(dataset.column_names) != {"input_ids", "attention_mask", "labels"}:
        raise ValueError("Unexpected model input columns")
    return dataset


def save_evaluation(split, output, rows, indices):
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    predictions = np.argmax(logits, axis=-1)
    labels = output.label_ids
    names = list(CATEGORIES.values())
    metrics = metric_values(labels, predictions)
    report = classification_report(labels, predictions, labels=list(range(6)), target_names=names,
                                   zero_division=0, output_dict=True)
    matrix = confusion_matrix(labels, predictions, labels=list(range(6)))
    write_json(RESULTS_DIR / f"{split}_metrics.json", {**metrics, "records": len(indices)})
    write_json(RESULTS_DIR / f"{split}_classification_report.json", report)
    with (RESULTS_DIR / f"{split}_per_class_metrics.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["project_label", "category", "precision", "recall", "f1", "support"])
        for label, name in CATEGORIES.items():
            item = report[name]
            writer.writerow([label, name, item["precision"], item["recall"], item["f1-score"], item["support"]])
    with (RESULTS_DIR / f"{split}_confusion_matrix.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_category / predicted_category"] + names)
        writer.writerows([[name] + matrix[i].tolist() for i, name in enumerate(names)])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(6), yticks=range(6), xticklabels=names, yticklabels=names,
           xlabel="Predicted category", ylabel="True category", title=f"DistilBERT: {split}")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{split}_confusion_matrix.png", dpi=180)
    plt.close(fig)
    with (RESULTS_DIR / f"{split}_predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset_row", "text_group_id", "true_project_label", "true_category",
                         "predicted_project_label", "predicted_category"])
        for index, label, prediction in zip(indices, labels, predictions):
            writer.writerow([index, rows[index]["text_group_id"], int(label) + 1, CATEGORIES[int(label) + 1],
                             int(prediction) + 1, CATEGORIES[int(prediction) + 1]])
    print(split, metrics, flush=True)


def train(rows, splits, plan, resume_checkpoint: str | None = None):
    """Only this explicitly invoked function loads a model or starts training."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, enable_full_determinism)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() > 1:
        raise RuntimeError("This script is configured for one CPU/GPU process")
    if (MODEL_DIR / "model.safetensors").exists() or (RESULTS_DIR / "test_metrics.json").exists():
        raise FileExistsError("Completed run already exists; refusing to overwrite model/test outputs")
    if resume_checkpoint:
        checkpoint = Path(resume_checkpoint).resolve()
        if not checkpoint.is_relative_to((MODEL_DIR / "checkpoints").resolve()) or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("Resume path must be a checkpoint from this new experiment")
    elif MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        raise FileExistsError("Unfinished model run exists; use --resume-from-checkpoint")
    enable_full_determinism(SEED)
    snapshot = base_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    datasets, indices = {}, {}
    for split in SPLITS:
        indices[split] = [i for i, value in enumerate(splits) if value == split]
        datasets[split] = model_dataset(rows, indices[split], tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(snapshot), local_files_only=True, num_labels=6,
        id2label={k - 1: v for k, v in CATEGORIES.items()},
        label2id={v: k - 1 for k, v in CATEGORIES.items()})
    model.config.project_id2label = {str(k): v for k, v in CATEGORIES.items()}
    weights = training_class_weights([int(rows[i]["final_label"]) - 1 for i in indices["train"]])
    expected_weights = list(plan["training_only_class_weights"].values())
    if not np.allclose(weights, expected_weights):
        raise ValueError("Training weights differ from the frozen plan")

    def compute_loss(outputs, labels, num_items_in_batch=None):
        return classification_loss(outputs.logits, labels, weights, model.training, num_items_in_batch)

    def compute_metrics(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        return metric_values(eval_prediction.label_ids, np.argmax(logits, axis=-1))

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"), num_train_epochs=HYPERPARAMETERS["epochs"],
        per_device_train_batch_size=HYPERPARAMETERS["train_batch_size"],
        per_device_eval_batch_size=HYPERPARAMETERS["eval_batch_size"],
        gradient_accumulation_steps=HYPERPARAMETERS["gradient_accumulation_steps"],
        learning_rate=HYPERPARAMETERS["learning_rate"], weight_decay=HYPERPARAMETERS["weight_decay"],
        warmup_steps=plan["warmup_steps"], lr_scheduler_type="linear", optim="adamw_torch",
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=2,
        seed=SEED, data_seed=SEED, full_determinism=True, fp16=False, bf16=False,
        dataloader_num_workers=0, dataloader_pin_memory=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(), logging_steps=50, report_to="none", push_to_hub=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=datasets["train"],
                      eval_dataset=datasets["validation"], processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics, compute_loss_func=compute_loss)
    # Trainer's custom-loss path already handles accumulation normalization;
    # num_items_in_batch is consumed by our loss, not passed into DistilBERT.
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=str(checkpoint) if resume_checkpoint else None)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    write_json(MODEL_DIR / "label_mapping.json", label_mapping())
    write_json(RESULTS_DIR / "training_history.json", trainer.state.log_history)
    write_json(RESULTS_DIR / "best_checkpoint.json", {"checkpoint": trainer.state.best_model_checkpoint,
                                                    "validation_macro_f1": trainer.state.best_metric})
    # Validation selects the best checkpoint. Test is evaluated once afterwards.
    for split in ["validation", "test"]:
        output = trainer.predict(datasets[split], metric_key_prefix=split)
        save_evaluation(split, output, rows, indices[split])
    if sha256_file(DATASET) != plan["dataset_sha256"]:
        raise RuntimeError("Dataset changed during training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="Prepare split/config only (default)")
    mode.add_argument("--train", action="store_true", help="Explicitly start DistilBERT training")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    if args.resume_from_checkpoint and not args.train:
        parser.error("--resume-from-checkpoint requires --train")
    rows, splits, plan = prepare()
    if args.train:
        train(rows, splits, plan, args.resume_from_checkpoint)
    else:
        print("Preparation complete. No model loaded, no training started, no network requests.")
        print(f"Frozen preparation outputs: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
