"""Run the existing DistilBERT configuration once on the audited correction.

Original model/results stay intact. Original per-row partitions are preserved.
No architecture, optimizer, hyperparameter, base revision, or seed is changed.
"""
import argparse
import json

import correct_native_ads as correction
import train_distilbert_six_category as experiment


def configure_experiment():
    experiment.MODEL_DIR = experiment.ROOT / "models/distilbert_six_category_native_ads_corrected"
    experiment.RESULTS_DIR = experiment.ROOT / "results/distilbert_six_category_native_ads_corrected"
    experiment.SPLIT_FILE = experiment.RESULTS_DIR / "split_manifest.csv"
    experiment.PLAN_FILE = experiment.RESULTS_DIR / "training_plan.json"
    experiment.MAPPING_FILE = experiment.RESULTS_DIR / "label_mapping.json"


def validate_correction():
    audit = json.loads((correction.AUDIT_DIR / "audit.json").read_text(encoding="utf-8"))
    if experiment.sha256_file(experiment.DATASET) != audit["corrected_dataset_sha256"]:
        raise ValueError("Corrected dataset differs from audited contents")
    if experiment.sha256_file(correction.BACKUP) != audit["original_dataset_sha256"]:
        raise ValueError("Original dataset backup differs from audit")
    _, before = correction.read_csv(correction.BACKUP)
    _, after = correction.read_csv(experiment.DATASET)
    if len(before) != len(after):
        raise ValueError("Dataset row count changed")
    for left, right in zip(before, after):
        allowed = {"text", "text_group_id", "selection_basis"} if left["final_label"] == "1" else set()
        if any(left[key] != right[key] for key in left if key not in allowed):
            raise ValueError("A protected category or provenance field changed")
    previous_plan = json.loads((correction.OLD_MANIFEST.parent / "training_plan.json").read_text(encoding="utf-8"))
    if previous_plan["hyperparameters"] != experiment.HYPERPARAMETERS:
        raise ValueError("Training hyperparameters changed")
    if previous_plan["base_revision"] != experiment.BASE_REVISION:
        raise ValueError("Base model revision changed")
    if experiment.sha256_file(correction.OLD_MANIFEST) != audit["source_manifest_sha256"]:
        raise ValueError("Original partition manifest changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Start the single authorized training run")
    args = parser.parse_args()
    configure_experiment()
    validate_correction()
    rows, splits, plan = experiment.prepare(reuse_split_manifest=correction.OLD_MANIFEST)
    _, previous = correction.read_csv(correction.OLD_MANIFEST)
    if splits != [row["split"] for row in previous]:
        raise ValueError("Original partition assignments were not preserved")
    if args.train:
        experiment.train(rows, splits, plan)
    else:
        print("Corrected dataset and unchanged training configuration verified. No training started.")


if __name__ == "__main__":
    main()
