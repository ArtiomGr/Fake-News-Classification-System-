"""Compare completed final-task experiments on the identical frozen partitions."""
import csv
from pathlib import Path
from final_project import ROOT, MODEL_PATHS, RESULT_PATHS, DISTIL_RESULTS, METRICS, frozen_data, read_json, sha256


def main():
    _, _, source = frozen_data()
    output = ROOT / "results/final_model_comparison"
    output.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        rows = []
        for name, folder in RESULT_PATHS.items():
            plan = read_json(folder / "training_plan.json")
            for key in ("dataset_sha256", "split_manifest_sha256"):
                if plan[key] != source[key]:
                    raise ValueError(f"{name} does not use the same final dataset/split")
            if sha256(folder / "split_manifest.csv") != source["split_manifest_sha256"]:
                raise ValueError(f"{name} split manifest changed")
            metrics = read_json(folder / f"{split}_metrics.json")
            if metrics["records"] != 1269:
                raise ValueError("Unexpected evaluation size")
            rows.append({"Model": name, **{key: metrics[key] for key in METRICS}})
        with (output / f"{split}_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Model", *METRICS])
            writer.writeheader()
            writer.writerows(rows)
        print(split, rows, flush=True)
    import json
    registry = {
        "dataset_sha256": source["dataset_sha256"], "split_manifest_sha256": source["split_manifest_sha256"],
        "models": {name: {"path": str(path), "weights_sha256": sha256(path / ("pipeline.joblib" if name == "Logistic Regression" else "model.safetensors")),
                          "mapping_sha256": sha256(path / "label_mapping.json")}
                   for name, path in MODEL_PATHS.items()}
    }
    (output / "model_registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
