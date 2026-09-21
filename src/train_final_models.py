"""Explicit offline BERT/LR experiments on the already-frozen final task.

Never retrains DistilBERT or changes data/splits. No hyperparameter search.
Existing completed new-model outputs are protected against accidental reruns.
"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

from final_project import (CATEGORIES, DATASET, DISTIL_RESULTS, MAPPING, MODEL_PATHS,
                           RESULT_PATHS, SPLIT_FILE, evaluate, frozen_data, sha256, write_json)


def start(model_name):
    rows, indices, source_plan = frozen_data()
    output, results = MODEL_PATHS[model_name], RESULT_PATHS[model_name]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing a second training run in {output}")
    if (results / "training_plan.json").exists():
        raise FileExistsError("An experiment already exists; inspect it rather than silently restarting")
    output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SPLIT_FILE, results / "split_manifest.csv")
    write_json(output / "label_mapping.json", MAPPING)
    write_json(results / "label_mapping.json", MAPPING)
    provenance = {"model": model_name, "dataset": str(DATASET), "dataset_sha256": sha256(DATASET),
                  "split_manifest": str(SPLIT_FILE), "split_manifest_sha256": sha256(SPLIT_FILE),
                  "split_totals": {key: len(value) for key, value in indices.items()},
                  "random_state": 42, "source_plan": str(DISTIL_RESULTS / "training_plan.json"),
                  "limitations": source_plan["limitations"],
                  "versions": {key: importlib.metadata.version(key) for key in ["torch", "transformers", "scikit-learn", "numpy"]}}
    return rows, indices, source_plan, output, results, provenance


def train_lr():
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    rows, indices, _, output, results, plan = start("Logistic Regression")
    plan["hyperparameters"] = {"max_features": 30000, "ngram_range": [1, 2], "min_df": 2,
                               "max_df": 0.95, "sublinear_tf": True, "stop_words": None,
                               "C": 1.0, "class_weight": "balanced", "solver": "lbfgs",
                               "max_iter": 2000, "tol": 0.0001, "random_state": 42}
    plan["feature_fitting"] = "Vocabulary and IDF fitted exclusively to the 5,921 frozen training rows; negation retained"
    write_json(results / "training_plan.json", plan)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2,
                                max_df=.95, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                                          max_iter=2000, tol=1e-4, random_state=42)),
    ])
    start_time = time.perf_counter()
    pipeline.fit([rows[i]["text"] for i in indices["train"]],
                 [int(rows[i]["final_label"]) - 1 for i in indices["train"]])
    if int(pipeline["classifier"].n_iter_.max()) >= 2000:
        raise RuntimeError("Baseline did not converge; do not report it as a converged experiment")
    assert pipeline.classes_.tolist() == list(range(6))
    joblib.dump(pipeline, output / "pipeline.joblib")
    write_json(output / "experiment.json", plan)
    write_json(results / "training_summary.json", {"seconds": time.perf_counter() - start_time,
                                                  "iterations": pipeline["classifier"].n_iter_.tolist(),
                                                  "vocabulary_size": len(pipeline["tfidf"].vocabulary_)})
    for split in ("validation", "test"):
        probabilities = pipeline.predict_proba([rows[i]["text"] for i in indices[split]])
        evaluate(split, probabilities, rows, indices[split], results)
    frozen_data()


def train_bert():
    import torch
    from datasets import Dataset
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, enable_full_determinism)
    from train_distilbert_six_category import classification_loss, metric_values, training_class_weights
    revision = "86b5e0934494bd15c9632b12f734a8a67f723594"
    snapshot = Path(HF_HUB_CACHE) / "models--bert-base-uncased/snapshots" / revision
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        if not (snapshot / name).is_file():
            raise FileNotFoundError(f"Original offline pretrained BERT asset missing: {snapshot / name}")
    rows, indices, source_plan, output, results, plan = start("BERT")
    hp = dict(source_plan["hyperparameters"])
    plan.update(base_model="bert-base-uncased", base_revision=revision, base_snapshot=str(snapshot),
                base_weights_sha256=sha256(snapshot / "model.safetensors"), hyperparameters=hp,
                warmup_steps=source_plan["warmup_steps"],
                training_only_class_weights=source_plan["training_only_class_weights"],
                initialization="Original pretrained BERT base, fresh six-category classifier; no old Fakeddit fine-tuned weights")
    write_json(results / "training_plan.json", plan)
    enable_full_determinism(42)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    datasets = {}
    for split, positions in indices.items():
        dataset = Dataset.from_dict({"text": [rows[i]["text"] for i in positions],
                                     "labels": [int(rows[i]["final_label"]) - 1 for i in positions]})
        datasets[split] = dataset.map(lambda batch: tokenizer(batch["text"], max_length=hp["max_length"],
                                       truncation=True, return_token_type_ids=False), batched=True,
                                       remove_columns=["text"], load_from_cache_file=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(snapshot), local_files_only=True,
                num_labels=6, id2label={key - 1: value for key, value in CATEGORIES.items()},
                label2id={value: key - 1 for key, value in CATEGORIES.items()})
    model.config.project_id2label = {str(key): value for key, value in CATEGORIES.items()}
    weights = training_class_weights([int(rows[i]["final_label"]) - 1 for i in indices["train"]])
    assert np.allclose(weights, list(plan["training_only_class_weights"].values()))
    args = TrainingArguments(output_dir=str(output / "checkpoints"), num_train_epochs=hp["epochs"],
             per_device_train_batch_size=hp["train_batch_size"], per_device_eval_batch_size=hp["eval_batch_size"],
             gradient_accumulation_steps=hp["gradient_accumulation_steps"], learning_rate=hp["learning_rate"],
             weight_decay=hp["weight_decay"], warmup_steps=plan["warmup_steps"], lr_scheduler_type="linear",
             optim="adamw_torch", eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
             metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=2,
             seed=42, data_seed=42, full_determinism=True, fp16=False, bf16=False,
             dataloader_num_workers=0, dataloader_pin_memory=torch.cuda.is_available(),
             use_cpu=not torch.cuda.is_available(), logging_steps=50, report_to="none", push_to_hub=False)
    trainer = Trainer(model=model, args=args, train_dataset=datasets["train"], eval_dataset=datasets["validation"],
              processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
              compute_loss_func=lambda outputs, labels, num_items_in_batch=None: classification_loss(
                  outputs.logits, labels, weights, model.training, num_items_in_batch),
              compute_metrics=lambda prediction: metric_values(prediction.label_ids, np.argmax(prediction.predictions, axis=-1)))
    trainer.model_accepts_loss_kwargs = False
    trainer.train()
    # The installed Trainer loads raw state dictionaries, whereas BERT's saved
    # legacy LayerNorm gamma/beta names need from_pretrained's key conversion.
    # Restore every parameter strictly before exporting or evaluating the best.
    best_model = AutoModelForSequenceClassification.from_pretrained(
        trainer.state.best_model_checkpoint, local_files_only=True)
    model.load_state_dict(best_model.state_dict(), strict=True)
    del best_model
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    write_json(output / "experiment.json", plan)
    write_json(results / "training_history.json", trainer.state.log_history)
    write_json(results / "best_checkpoint.json", {"checkpoint": trainer.state.best_model_checkpoint,
                                                  "validation_macro_f1": trainer.state.best_metric})
    for split in ("validation", "test"):
        predictions = trainer.predict(datasets[split])
        # Argmax and metrics do not require converting logits into probabilities.
        evaluate(split, predictions.predictions, rows, indices[split], results)
    frozen_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["bert", "logistic"], required=True)
    args = parser.parse_args()
    (train_bert if args.model == "bert" else train_lr)()
