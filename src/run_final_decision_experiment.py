"""Reproducible phases 1-5. Only new v1 result/model directories are written.

Run freeze, classifiers, fusion, then report. Classifier weights are never fit.
Use -B and a runtime outside the protected .venv. Completed stages cannot rerun.
"""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, ConfusionMatrixDisplay)
from sklearn.model_selection import StratifiedGroupKFold

from final_project import (ROOT, DATASET, SPLIT_FILE, MODEL_PATHS, RESULT_PATHS,
                           CATEGORIES, MAPPING, frozen_data, read_json, sha256, write_json)
from final_decision import (ARTIFACT_DIR, PROBABILITY_FEATURES, SENTIMENT_FEATURES,
                            SYSTEMS, decision_features, make_estimator, probability_matrix,
                            vader_features)

OUT = ROOT / "results/final_decision_v1"
SLUGS = {"BERT": "bert", "DistilBERT": "distilbert", "Logistic Regression": "logistic_regression"}
GRID = [.01, .1, 1., 10., 100.]
SEED = 42
LIMITATIONS = [
    "Validation records influenced frozen classifier checkpoint selection; nested decision-layer CV does not undo this dependence.",
    "Earlier test/probe results informed Native Advertising dataset correction; test is locked regression evidence, not untouched independent evidence.",
    "Source-supervised/heuristic labels and source/style correlations limit generalization; sources are not held out wholesale.",
    "Known groups are separated; undetected semantic overlap may remain.",
    "Only 144 validation and 144 test Fabrication examples; small differences can be unstable.",
    "Final decision layers fit validation only. Their train-partition scores are diagnostics on base-classifier training data, not generalization estimates.",
    "Final-fit validation scores are resubstitution; nested out-of-fold scores are the decision development comparison.",
    "No independent long-document accuracy benchmark; runtime uses existing overlapping-window mean probabilities.",
    "Probabilities are assessed for calibration but are not factual verification or guaranteed calibrated confidence.",
]


def csv_write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, mode="x")


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def protect_paths():
    paths = [ROOT / f["path"] for f in read_json(ROOT / "models/artifact_manifest.json")["files"]]
    paths += [DATASET, ROOT / "models/artifact_manifest.json", ROOT / "app/app.py", ROOT / "README.md"]
    for directory in [*RESULT_PATHS.values(), ROOT / "results/final_model_comparison"]:
        paths.extend(p for p in directory.iterdir() if p.is_file())
    return sorted(set(paths))


def verify_protected():
    snapshot = read_json(OUT / "protected_inputs.json")
    for relative, expected in snapshot.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Protected input changed: {relative}")
    return len(snapshot)


def freeze():
    if OUT.exists() or ARTIFACT_DIR.exists():
        raise FileExistsError("Versioned experiment already exists; never overwrite it")
    rows, indices, plan = frozen_data()
    for item in read_json(ROOT / "models/artifact_manifest.json")["files"]:
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Artifact integrity failure: {path}")
    for name, path in RESULT_PATHS.items():
        if sha256(path / "split_manifest.csv") != plan["split_manifest_sha256"]:
            raise ValueError(f"Split mismatch: {name}")
    OUT.mkdir(parents=True)
    snapshot = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in protect_paths()}
    write_json(OUT / "protected_inputs.json", snapshot)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "experiment": "final_decision_v1",
        "dataset_sha256": plan["dataset_sha256"], "split_manifest_sha256": plan["split_manifest_sha256"],
        "label_mapping": MAPPING, "split_counts": {k: len(v) for k, v in indices.items()},
        "classifier_selection": "Highest saved validation macro F1; exact tie follows BERT, DistilBERT, Logistic Regression order. Test never selects.",
        "decision_selection": "Highest nested validation OOF macro F1 among A/B/C; exact ties prefer A then B then C. C must exceed both A and B.",
        "development": "Only existing validation partition; nested StratifiedGroupKFold (outer 5, inner 4), random_state=42. Inner seed=42+outer_fold. Final refit C chosen by grouped 5-fold CV on validation.",
        "regularization_C_grid": GRID, "tuning_metric": "mean fold macro F1", "tuning_tie_break": "smaller C",
        "estimator": "StandardScaler then multinomial LogisticRegression, L2, lbfgs, max_iter=3000, tol=1e-8, no class weights",
        "feature_order": {"probability_only": list(PROBABILITY_FEATURES), "vader_fusion": list(PROBABILITY_FEATURES + SENTIMENT_FEATURES)},
        "seed": SEED, "outer_folds": 5, "inner_folds": 4,
        "inference": "Whitespace normalization; float32 Transformers in eval/inference mode; 512 tokens, no token_type_ids, dynamic padding batch=16; assert dataset has no >512-token records. Runtime long text retains 16-token overlap and mean window probabilities. LR full normalized text.",
        "torch_cpu_threads": 4, "brier_definition": "mean over rows of sum over six classes (p - one_hot(y))^2; range 0..2",
        "uncertainty": "2000 paired cluster bootstrap resamples of related group IDs; percentile 95% intervals, seed 42; conditional on fitted models, not retraining uncertainty",
        "vader_measurable_improvement_rule": "C exceeds A and B in test macro F1 and both paired 95% cluster-bootstrap macro-F1 difference intervals exclude zero; development selection remains fixed regardless.",
        "numeric_product_acceptance_targets": None,
        "acceptance_note": "No accuracy or response-time thresholds supplied; report measurements without inventing pass/fail targets.",
        "limitations": LIMITATIONS,
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "scikit-learn", "numpy", "pandas", "vaderSentiment", "joblib", "matplotlib"]},
        "platform": platform.platform(), "python": platform.python_version(),
    }
    write_json(OUT / "experiment_config.json", config)
    validation = {name: read_json(path / "validation_metrics.json")["macro_f1"] for name, path in RESULT_PATHS.items()}
    selected = max(validation, key=validation.get)
    write_json(OUT / "classifier_selection.json", {"selected_classifier": selected,
               "criterion": "validation macro F1", "validation_macro_f1": validation,
               "test_used_for_selection": False})
    print("Frozen", len(snapshot), "protected files; selected", selected, flush=True)


def metrics(y, probabilities):
    p = probability_matrix(probabilities)
    pred = p.argmax(axis=1)
    return {"records": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=range(6), zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", labels=range(6), zero_division=0)),
            "log_loss": float(log_loss(y, p, labels=list(range(6)))),
            "multiclass_brier": float(np.mean(np.sum((p - np.eye(6)[y]) ** 2, axis=1)))}


def save_evaluation(folder, split, ids, rows, probabilities):
    folder.mkdir(parents=True, exist_ok=True)
    y = np.array([int(rows[i]["final_label"]) - 1 for i in ids])
    p = probability_matrix(probabilities)
    pred = p.argmax(axis=1)
    result = metrics(y, p)
    write_json(folder / f"{split}_metrics.json", result)
    report = classification_report(y, pred, labels=list(range(6)), target_names=list(CATEGORIES.values()),
                                   output_dict=True, zero_division=0)
    write_json(folder / f"{split}_classification_report.json", report)
    csv_write(folder / f"{split}_per_class_metrics.csv", pd.DataFrame([
        {"project_label": i, "category": label, **report[label]} for i, label in CATEGORIES.items()]))
    matrix = confusion_matrix(y, pred, labels=list(range(6)))
    frame = pd.DataFrame(matrix, columns=CATEGORIES.values())
    frame.insert(0, "true_category", list(CATEGORIES.values()))
    csv_write(folder / f"{split}_confusion_matrix.csv", frame)
    frame = pd.DataFrame({"dataset_row": ids, "true_project_label": y + 1,
                          "predicted_project_label": pred + 1, "confidence": p.max(axis=1)})
    for j, feature in enumerate(PROBABILITY_FEATURES):
        frame[feature] = p[:, j]
    csv_write(folder / f"{split}_predictions.csv", frame)
    plt = plotting()
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(matrix, display_labels=list(CATEGORIES.values())).plot(ax=ax, xticks_rotation=35, colorbar=False)
    ax.set_title(f"{folder.name}: {split}")
    fig.tight_layout(); fig.savefig(folder / f"{split}_confusion_matrix.png", dpi=150); plt.close(fig)
    return result


def input_frame(rows):
    frame = pd.read_csv(SPLIT_FILE, keep_default_na=False)
    frame["text_sha256"] = [__import__("hashlib").sha256(r["text"].encode("utf-8")).hexdigest() for r in rows]
    return frame


def classifier_probabilities(name, rows):
    import torch
    from predict import load_classifier, predict_model, _load_classifier
    torch.set_num_threads(4)
    tokenizer, model = load_classifier(name)
    texts = [" ".join(r["text"].split()) for r in rows]
    started = time.perf_counter()
    if tokenizer is None:
        raw = model.predict_proba(texts)
        probs = np.zeros((len(rows), 6))
        probs[:, model.classes_.astype(int)] = raw
        lengths = None
    else:
        encoded = tokenizer(texts, truncation=False, return_token_type_ids=False)
        lengths = list(map(len, encoded["input_ids"]))
        if max(lengths) > 512:
            raise ValueError("Dataset exceeds frozen evaluation policy; do not silently truncate or change it")
        probs = np.empty((len(rows), 6), dtype=np.float64)
        for start in range(0, len(rows), 16):
            batch = tokenizer.pad({k: v[start:start+16] for k, v in encoded.items()}, padding=True, return_tensors="pt")
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.inference_mode():
                probs[start:start+16] = model(**batch).logits.softmax(dim=-1).cpu().numpy()
            if start % 512 == 0:
                print(name, min(start + 16, len(rows)), "/", len(rows), flush=True)
    elapsed = time.perf_counter() - started
    parity = []
    # Fixed predeclared positions spanning every partition; no outcome-based sampling.
    for i in [0, 1, 6, 42, 100, 1000, 4000, 8458]:
        runtime = predict_model(texts[i], name)
        expected = np.array(list(runtime["probability_by_label"].values()))
        error = float(np.max(np.abs(expected - probs[i])))
        if error > 1e-5:
            raise ValueError(f"Batched/runtime probability mismatch: {name}, row {i}, {error}")
        parity.append({"dataset_row": i, "max_absolute_probability_difference": error})
    info = {"seconds": elapsed, "records": len(rows), "device": str(model.device) if tokenizer else "cpu",
            "max_token_length": max(lengths) if lengths else None, "runtime_parity": parity}
    _load_classifier.cache_clear()
    return probability_matrix(probs), info


def classifiers():
    verify_protected()
    rows, indices, _ = frozen_data()
    records = []
    for name, slug in SLUGS.items():
        folder = OUT / "classifiers" / slug
        done = folder / "completed.json"
        if done.exists():
            if sha256(folder / "probabilities.csv") != read_json(done)["probabilities_sha256"]:
                raise ValueError("Completed probabilities changed")
        else:
            if folder.exists():
                raise FileExistsError(f"Incomplete output needs inspection: {folder}")
            probabilities, info = classifier_probabilities(name, rows)
            frame = input_frame(rows)
            for j, feature in enumerate(PROBABILITY_FEATURES): frame[feature] = probabilities[:, j]
            csv_write(folder / "probabilities.csv", frame)
            for split, ids in indices.items():
                result = save_evaluation(folder, split, ids, rows, probabilities[ids])
                if split != "train":
                    old = pd.read_csv(RESULT_PATHS[name] / f"{split}_predictions.csv")
                    actual = dict(zip(ids, probabilities[ids].argmax(axis=1) + 1))
                    mismatches = sum(actual[int(r.dataset_row)] != int(r.predicted_project_label) for r in old.itertuples())
                    if mismatches or len(old) != len(ids):
                        raise ValueError(f"Historical prediction mismatch: {name}/{split}: {mismatches}")
                    reference = read_json(RESULT_PATHS[name] / f"{split}_metrics.json")
                    for key in ("accuracy", "macro_f1", "weighted_f1"):
                        if abs(result[key] - reference[key]) > 1e-12:
                            raise ValueError(f"Metric parity failure: {name}/{split}/{key}")
            info.update(probabilities_sha256=sha256(folder / "probabilities.csv"), historical_validation_test_predictions_match=True)
            write_json(done, info)
        for split in indices:
            records.append({"model": name, "split": split, **read_json(folder / f"{split}_metrics.json")})
    csv_write(OUT / "classifier_comparison.csv", pd.DataFrame(records))
    gaps = []
    for name in SLUGS:
        by_split = {r["split"]: r for r in records if r["model"] == name}
        gaps.append({"model": name, **{f"train_minus_{s}_{m}": by_split["train"][m] - by_split[s][m]
                   for s in ("validation", "test") for m in ("accuracy", "macro_f1", "weighted_f1")}})
    csv_write(OUT / "generalization_gaps.csv", pd.DataFrame(gaps))
    comparison_graphs(pd.DataFrame(records), "model", "classifier")
    write_json(OUT / "classifiers_completed.json", {"protected_files_verified": verify_protected()})


def comparison_graphs(frame, column, prefix):
    plt = plotting()
    for metric in ("accuracy", "macro_f1"):
        table = frame.pivot(index=column, columns="split", values=metric)
        ordered = [s for s in ("train", "validation", "validation_nested_oof", "test") if s in table]
        ax = table[ordered].plot.bar(figsize=(11, 5), ylim=(0, 1.03), rot=0)
        ax.set_ylabel(metric.replace("_", " ")); ax.set_xlabel("")
        ax.set_title(f"{prefix}: {metric.replace('_', ' ')}")
        ax.legend(title="partition"); ax.figure.tight_layout()
        ax.figure.savefig(OUT / f"{prefix}_{metric}_comparison.png", dpi=160); plt.close(ax.figure)


def group_folds(y, groups, n_splits, seed):
    folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    coverage = np.zeros(len(y), dtype=int)
    for train, held in folds:
        if set(groups[train]) & set(groups[held]): raise ValueError("Group leakage")
        if set(y[train]) != set(range(6)) or set(y[held]) != set(range(6)):
            raise ValueError("Every development fold must contain all six categories")
        coverage[held] += 1
    if not np.all(coverage == 1): raise ValueError("Fold coverage failure")
    return folds


def tune(features, y, folds):
    results = []
    for c in GRID:
        scores = []
        for train, held in folds:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model = make_estimator(c).fit(features[train], y[train])
            scores.append(float(f1_score(y[held], model.predict(features[held]), average="macro", labels=range(6), zero_division=0)))
        results.append({"C": c, "fold_macro_f1": scores, "mean_macro_f1": float(np.mean(scores))})
    best = max(results, key=lambda r: (r["mean_macro_f1"], -r["C"]))
    return best["C"], results


def paired_bootstrap(y, left, right, groups):
    """Cluster bootstrap from group confusion matrices; preserve paired predictions."""
    unique = np.unique(groups)
    matrices = []
    for p in (left, right):
        pred = p.argmax(axis=1)
        matrices.append(np.stack([confusion_matrix(y[groups == g], pred[groups == g], labels=range(6)) for g in unique]))
    def measures(cm):
        tp = np.diag(cm).astype(float); actual = cm.sum(axis=1); predicted = cm.sum(axis=0)
        f1 = np.divide(2*tp, actual+predicted, out=np.zeros(6), where=(actual+predicted) != 0)
        return np.array([tp.sum()/cm.sum(), f1.mean(), np.sum(f1*actual)/cm.sum()])
    rng = np.random.default_rng(SEED)
    diffs = np.empty((2000, 3))
    for i in range(len(diffs)):
        sample = rng.integers(0, len(unique), len(unique))
        diffs[i] = measures(matrices[0][sample].sum(axis=0)) - measures(matrices[1][sample].sum(axis=0))
    observed = measures(matrices[0].sum(axis=0)) - measures(matrices[1].sum(axis=0))
    return {key: {"difference": float(observed[j]), "ci95_low": float(np.quantile(diffs[:, j], .025)),
                  "ci95_high": float(np.quantile(diffs[:, j], .975))}
            for j, key in enumerate(("accuracy", "macro_f1", "weighted_f1"))}


def fusion():
    import joblib
    verify_protected()
    if not (OUT / "classifiers_completed.json").exists(): raise ValueError("Evaluate classifiers first")
    if ARTIFACT_DIR.exists() or (OUT / "fusion").exists(): raise FileExistsError("Fusion already started; do not overwrite")
    rows, indices, _ = frozen_data()
    selected = read_json(OUT / "classifier_selection.json")["selected_classifier"]
    source = OUT / "classifiers" / SLUGS[selected] / "probabilities.csv"
    if sha256(source) != read_json(source.parent / "completed.json")["probabilities_sha256"]:
        raise ValueError("Selected classifier probability export changed after verification")
    frame = pd.read_csv(source, keep_default_na=False)
    expected = input_frame(rows)
    for column in expected.columns:
        if frame[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError(f"Feature row/provenance mismatch: {column}")
    p = probability_matrix(frame[list(PROBABILITY_FEATURES)].to_numpy())
    sentiment = vader_features([r["text"] for r in rows])
    for j, feature in enumerate(SENTIMENT_FEATURES): frame[feature] = sentiment[:, j]
    csv_write(OUT / "fusion_features.csv", frame)
    write_json(OUT / "feature_provenance.json", {"selected_classifier": selected,
        "source_probabilities_sha256": sha256(source), "features_sha256": sha256(OUT / "fusion_features.csv"),
        "feature_names": list(PROBABILITY_FEATURES + SENTIMENT_FEATURES),
        "text": "Exact dataset text, with identical whitespace normalization for classifier and VADER; text_sha256 records original UTF-8 text.",
        "metadata_not_model_inputs": ["dataset_row", "split", "group_id", "text_group_id", "project_label", "internal_label", "text_sha256"]})
    dev = np.array(indices["validation"])
    y_all = np.array([int(r["final_label"])-1 for r in rows])
    y = y_all[dev]; groups = frame.group_id.to_numpy()[dev]
    outer = group_folds(y, groups, 5, SEED)
    fold_records = []
    for number, (train, held) in enumerate(outer):
        for role, positions in (("fit", train), ("heldout", held)):
            fold_records.extend({"outer_fold": number, "inner_fold": -1, "role": role,
                                 "dataset_row": int(dev[i]), "group_id": groups[i]} for i in positions)
        inner = group_folds(y[train], groups[train], 4, SEED+number)
        for inner_number, (fit, valid) in enumerate(inner):
            for role, positions in (("fit", fit), ("heldout", valid)):
                fold_records.extend({"outer_fold": number, "inner_fold": inner_number, "role": role,
                                     "dataset_row": int(dev[train[i]]), "group_id": groups[train[i]]} for i in positions)
    final_folds = group_folds(y, groups, 5, SEED)
    for number, (train, held) in enumerate(final_folds):
        for role, positions in (("fit", train), ("heldout", held)):
            fold_records.extend({"outer_fold": -1, "inner_fold": number, "role": role,
                                 "dataset_row": int(dev[i]), "group_id": groups[i]} for i in positions)
    csv_write(OUT / "development_folds.csv", pd.DataFrame(fold_records))
    ARTIFACT_DIR.mkdir(parents=True)
    artifacts = {}; oof = {"classifier_alone": p[dev]}; fitted = {}; tuning = {}
    for system in SYSTEMS[1:]:
        x = decision_features(p, sentiment if system == "vader_fusion" else None)[dev]
        predicted = np.empty((len(dev), 6)); details = []
        for number, (train, held) in enumerate(outer):
            c, search = tune(x[train], y[train], group_folds(y[train], groups[train], 4, SEED+number))
            model = make_estimator(c).fit(x[train], y[train])
            predicted[held] = model.predict_proba(x[held])
            details.append({"outer_fold": number, "selected_C": c, "inner_search": search})
        c, search = tune(x, y, final_folds)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted[system] = make_estimator(c).fit(x, y)
        oof[system] = probability_matrix(predicted)
        tuning[system] = {"nested_folds": details, "final_search": search, "final_C": c}
        filename = f"{system}.joblib"
        joblib.dump(fitted[system], ARTIFACT_DIR / filename)
        artifacts[system] = {"filename": filename, "sha256": sha256(ARTIFACT_DIR / filename), "C": c,
            "feature_names": list(PROBABILITY_FEATURES + (SENTIMENT_FEATURES if system == "vader_fusion" else ())),
            "fit_partition": "validation", "fit_records": len(dev), "classes": list(range(6))}
        scaler, logistic = fitted[system].steps[0][1], fitted[system].steps[1][1]
        write_json(ARTIFACT_DIR / f"{system}_parameters.json", {
            "feature_names": artifacts[system]["feature_names"], "classes": logistic.classes_.tolist(),
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coefficients": logistic.coef_.tolist(), "intercepts": logistic.intercept_.tolist(),
            "iterations": logistic.n_iter_.tolist(), "C": c,
            "formula": "softmax(((features - scaler_mean) / scaler_scale) @ coefficients.T + intercepts)"})
    write_json(OUT / "tuning_results.json", tuning)
    development_metrics = {system: metrics(y, values) for system, values in oof.items()}
    recommended = max(SYSTEMS, key=lambda s: (development_metrics[s]["macro_f1"], -SYSTEMS.index(s)))
    registry = read_json(ROOT / "results/final_model_comparison/model_registry.json")
    # Persist deployment selection BEFORE accessing test outcomes for A/B/C.
    deployment = {"schema_version": 1, "selected_classifier": selected,
        "selected_classifier_weights_sha256": registry["models"][selected]["weights_sha256"],
        "recommended_system": recommended, "selection_basis": "nested validation OOF macro F1; ties prefer A, B, C",
        "development_metrics": development_metrics, "test_used_for_selection": False,
        "label_mapping": MAPPING, "artifacts": artifacts,
        "dataset_sha256": sha256(DATASET), "split_manifest_sha256": sha256(SPLIT_FILE),
        "experiment_config_sha256": sha256(OUT / "experiment_config.json"), "limitations": LIMITATIONS}
    write_json(ARTIFACT_DIR / "deployment.json", deployment)
    print("Development-selected system:", recommended, development_metrics, flush=True)
    all_predictions = {"classifier_alone": p}
    for system in SYSTEMS[1:]:
        all_predictions[system] = probability_matrix(fitted[system].predict_proba(
            decision_features(p, sentiment if system == "vader_fusion" else None)))
    comparison = []
    for system in SYSTEMS:
        folder = OUT / "fusion" / system
        result = save_evaluation(folder, "validation_nested_oof", dev, rows, oof[system])
        comparison.append({"system": system, "split": "validation_nested_oof", **result})
        for split, ids in indices.items():
            result = save_evaluation(folder, split, ids, rows, all_predictions[system][ids])
            comparison.append({"system": system, "split": split, **result})
    csv_write(OUT / "decision_comparison.csv", pd.DataFrame(comparison))
    changes = []; uncertainty = {}
    pairs = [("probability_only", "classifier_alone"), ("vader_fusion", "classifier_alone"), ("vader_fusion", "probability_only")]
    for split in ("validation_nested_oof", "test"):
        ids = dev if split == "validation_nested_oof" else np.array(indices["test"])
        predictions = oof if split == "validation_nested_oof" else {s: v[ids] for s, v in all_predictions.items()}
        uncertainty[split] = {}
        for left, right in pairs:
            lp, rp = predictions[left].argmax(axis=1), predictions[right].argmax(axis=1)
            key = left + "_minus_" + right
            uncertainty[split][key] = paired_bootstrap(y_all[ids], predictions[left], predictions[right], frame.group_id.to_numpy()[ids])
            for local in np.where(lp != rp)[0]:
                i = ids[local]
                changes.append({"split": split, "comparison": key, "dataset_row": int(i), "group_id": frame.group_id.iloc[i],
                    "true_project_label": int(y_all[i]+1), "left_project_label": int(lp[local]+1), "right_project_label": int(rp[local]+1),
                    "left_correct": bool(lp[local] == y_all[i]), "right_correct": bool(rp[local] == y_all[i])})
    csv_write(OUT / "changed_predictions.csv", pd.DataFrame(changes, columns=["split", "comparison", "dataset_row", "group_id", "true_project_label", "left_project_label", "right_project_label", "left_correct", "right_correct"]))
    write_json(OUT / "paired_uncertainty.json", uncertainty)
    measurable = all(uncertainty["test"]["vader_fusion_minus_"+s]["macro_f1"]["ci95_low"] > 0
                     and uncertainty["test"]["vader_fusion_minus_"+s]["macro_f1"]["difference"] > 0 for s in SYSTEMS[:2])
    write_json(OUT / "sentiment_value.json", {"measurable_incremental_macro_f1_benefit": measurable,
        "rule": read_json(OUT / "experiment_config.json")["vader_measurable_improvement_rule"],
        "recommended_system_from_development": recommended, "test_differences": uncertainty["test"],
        "independent_confirmation": "Not available; all test comparisons are regression evidence"})
    comparison_graphs(pd.DataFrame(comparison).query("split in ['validation_nested_oof','test']"), "system", "decision")
    calibration_graphs(y_all[np.array(indices["test"])], {s: v[indices["test"]] for s, v in all_predictions.items()})
    write_json(OUT / "fusion_completed.json", {"protected_files_verified": verify_protected(),
        "deployment_sha256": sha256(ARTIFACT_DIR / "deployment.json")})


def calibration_graphs(y, values):
    plt = plotting(); fig, axes = plt.subplots(1, 2, figsize=(12, 5)); records = []
    for system, p in values.items():
        confidence = p.max(axis=1); correct = p.argmax(axis=1) == y; xs = []; ys = []
        for i in range(10):
            mask = (confidence >= i/10) & ((confidence < (i+1)/10) if i < 9 else (confidence <= 1))
            if mask.any():
                xs.append(float(confidence[mask].mean())); ys.append(float(correct[mask].mean()))
                records.append({"system": system, "bin": i, "records": int(mask.sum()), "mean_confidence": xs[-1], "accuracy": ys[-1]})
        axes[0].plot(xs, ys, marker="o", label=system)
    axes[0].plot([0, 1], [0, 1], "k--"); axes[0].set(xlabel="Mean confidence", ylabel="Accuracy", title="Test reliability (regression evidence)")
    axes[0].legend()
    pd.DataFrame({s: {k: v for k, v in metrics(y, p).items() if k in ("log_loss", "multiclass_brier")} for s, p in values.items()}).T.plot.bar(ax=axes[1], rot=15)
    axes[1].set_title("Probability quality: lower is better")
    fig.tight_layout(); fig.savefig(OUT / "decision_probability_quality.png", dpi=160); plt.close(fig)
    csv_write(OUT / "test_reliability_bins.csv", pd.DataFrame(records))


def report():
    verify_protected()
    classifier = pd.read_csv(OUT / "classifier_comparison.csv")
    decision = pd.read_csv(OUT / "decision_comparison.csv")
    deployment = read_json(ARTIFACT_DIR / "deployment.json")
    sentiment = read_json(OUT / "sentiment_value.json")
    def table(frame, name):
        keys = [name, "split", "accuracy", "macro_f1", "weighted_f1", "log_loss", "multiclass_brier"]
        lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"]*len(keys)) + " |"]
        for record in frame[keys].to_dict("records"):
            lines.append("| " + " | ".join(f"{record[k]:.6f}" if isinstance(record[k], float) else str(record[k]) for k in keys) + " |")
        return "\n".join(lines)
    content = "\n\n".join([
        "# Final decision experiment v1 — phases 1–5",
        "Existing weights, mappings, dataset, splits, historical evaluation and Streamlit are preserved. Only the two new logistic decision layers were trained.",
        f"Selected classifier: **{deployment['selected_classifier']}**, by fixed validation macro F1. Recommended system: **{deployment['recommended_system']}**, by nested grouped validation OOF macro F1 before test comparison.",
        "## Frozen classifier results", table(classifier, "model"),
        "Validation/test hard predictions and metrics exactly reproduce historical final results. TRAIN results use the complete frozen partition, evaluation mode and the same inference definitions.",
        "## Decision comparison", table(decision, "system"),
        "Validation is final-layer resubstitution. validation_nested_oof is the honest decision-layer development estimate, subject to prior base-model selection dependence. Train is diagnostic because the base classifier already saw those rows.",
        "## VADER incremental value", json.dumps(sentiment, indent=2),
        "## Architecture", "Six ordered frozen classifier probabilities; optional [negative, neutral, positive, compound] VADER scores; fold-fitted StandardScaler; L2 multinomial LogisticRegression; six final probabilities; argmax category and maximum probability. Probability-only control differs only by omission of the four VADER features. Each system uses identical folds and C grid.",
        "## Provenance and evaluation", "See experiment_config.json, protected_inputs.json, development_folds.csv, feature_provenance.json, tuning_results.json, paired_uncertainty.json and changed_predictions.csv. Paired intervals resample complete related groups; they do not include model-training uncertainty. Classifier probabilities and VADER features are saved for all 8,459 records; no training-row probabilities fit a decision layer.",
        "## Limitations", "\n".join("- " + item for item in LIMITATIONS),
        "No numeric product accuracy/latency thresholds were supplied. Measurements do not establish acceptance against unspecified requirements. Streamlit integration is intentionally deferred.",
        "## Runtime interface", "`from final_decision import predict_final`\n\n`predict_final(text)` follows the recorded development selection. `predict_final(text, system='vader_fusion')` explicitly exercises the evaluated 10-feature fusion. No silent fallback occurs if sentiment or artifacts fail.",
    ])
    if (OUT / "test_results.json").exists():
        content += "\n\n## Complete test suite\n\n```json\n" + json.dumps(read_json(OUT / "test_results.json"), indent=2) + "\n```"
    if (OUT / "runtime_benchmark.json").exists():
        timing = read_json(OUT / "runtime_benchmark.json")
        content += "\n\n## API latency\n\n" + timing["scope"] + "\n\n| System | Input | Characters | p50 s | p95 s |\n| --- | --- | ---: | ---: | ---: |\n"
        content += "\n".join(f"| {r['system']} | {r['input_size']} | {r['characters']} | {r['p50_seconds']:.4f} | {r['p95_seconds']:.4f} |" for r in timing["warm_end_to_end"])
        content += "\n\nCold starts: " + json.dumps(timing["cold_start"]) + "\n\nSee edge_cases.json for the separate input robustness checks."
    content += "\n\n## Interpretation and integration boundary\n\n"
    content += ("The fixed development criterion selects corrected DistilBERT followed by the probability-only logistic decision layer. "
                "VADER has no demonstrated incremental predictive benefit over both controls: its development macro F1 is lower than the probability-only control, "
                "and both test paired macro-F1 difference intervals include zero. The small positive test point differences do not justify changing the frozen selection. "
                "Both learned layers worsen test log loss substantially relative to classifier-alone, and neither improves test Brier score. "
                "The chosen architecture follows the approved macro-F1 criterion; this is not evidence of improved probability calibration. "
                "The optional VADER fusion remains available for explicit experiments, not as the default. Streamlit and the presentation remain unchanged.")
    if (OUT / "edge_cases.json").exists():
        content += "\n\n## Edge cases\n\n```json\n" + json.dumps(read_json(OUT / "edge_cases.json"), indent=2) + "\n```"
    if (OUT / "integrity_verification.json").exists():
        content += "\n\n## Protected-file integrity\n\n```json\n" + json.dumps(read_json(OUT / "integrity_verification.json"), indent=2) + "\n```"
    content += "\n\n## Complete file and graph index\n\nSee [files_created_or_changed.txt](files_created_or_changed.txt) for every created or changed file and [file_inventory.csv](file_inventory.csv) for sizes and SHA256 hashes. The inventory includes prior completed outputs reused in this continuation.\n\n"
    content += "\n".join(f"- [{p.relative_to(OUT).as_posix()}]({p.relative_to(OUT).as_posix()})" for p in sorted(OUT.rglob('*.png')))
    content += "\n\n## Per-class results\n\nAll precision, recall and F1 values below are fractions. Confusion matrices and row-level probabilities are saved beside each table.\n\n"
    for path in sorted(OUT.rglob('*_per_class_metrics.csv')):
        frame = pd.read_csv(path)
        content += f"\n\n### {path.relative_to(OUT).as_posix()}\n\n"
        content += "| Category | Precision | Recall | F1 | Support |\n| --- | ---: | ---: | ---: | ---: |\n"
        content += "\n".join(f"| {r['category']} | {r['precision']:.6f} | {r['recall']:.6f} | {r['f1-score']:.6f} | {int(r['support'])} |" for r in frame.to_dict('records'))
    content += "\n\n## Execution notes\n\n" + json.dumps(read_json(OUT / 'execution_notes.json'), indent=2)
    (OUT / "experiment_report.md").write_text(content + "\n", encoding="utf-8")
    print(content, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "classifiers", "fusion", "report"))
    args = parser.parse_args()
    globals()[args.stage]()
