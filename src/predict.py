"""Final three-model inference. All resources are local and task-validated."""
from functools import lru_cache
import json
import os
from threading import RLock
from time import perf_counter

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
from final_project import ROOT, MODEL_PATHS, DISTIL_RESULTS, MAPPING, read_json, sha256

PROJECT_ROOT = ROOT
DISTILBERT_DIR = MODEL_PATHS["DistilBERT"]
MODEL_NAMES = tuple(MODEL_PATHS)
TRANSFORMER_MAX_LENGTH = 512
CHUNK_OVERLAP_TOKENS = 16
MAX_INPUT_CHARACTERS = 50_000
_INFERENCE_LOCK = RLock()


def artifact_fingerprint():
    """Cache identity includes exact paths and artifact versions, not just a name."""
    signature = []
    for name, path in MODEL_PATHS.items():
        for filename in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                         "label_mapping.json", "pipeline.joblib", "experiment.json"):
            file = path / filename
            stat = file.stat() if file.is_file() else None
            signature.append((name, str(file.resolve()), stat.st_mtime_ns if stat else None, stat.st_size if stat else None))
    registry = ROOT / "results/final_model_comparison/model_registry.json"
    stat = registry.stat() if registry.is_file() else None
    signature.append(("registry", str(registry), stat.st_mtime_ns if stat else None, stat.st_size if stat else None))
    return tuple(signature)


def get_model_metadata(model_name="DistilBERT"):
    path = MODEL_PATHS[model_name]
    required = ["pipeline.joblib", "experiment.json", "label_mapping.json"] if model_name == "Logistic Regression" else [
        "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "label_mapping.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Final local {model_name} model is incomplete at {path}: {', '.join(missing)}")
    mapping = read_json(path / "label_mapping.json")
    if mapping != MAPPING:
        raise ValueError(f"{model_name} mapping does not match the final project task")
    projects = {int(k): v for k, v in mapping["project_id_to_category"].items()}
    internal = {int(k): v for k, v in mapping["internal_id_to_category"].items()}
    to_project = {int(k): int(v) for k, v in mapping["internal_id_to_project_id"].items()}
    if set(projects) != set(range(1, 7)) or set(internal) != set(range(6)):
        raise ValueError("Expected exactly six final project categories")
    source_plan = read_json(DISTIL_RESULTS / "training_plan.json")
    if model_name != "DistilBERT":
        experiment = read_json(path / "experiment.json")
        if any(experiment[key] != source_plan[key] for key in ("dataset_sha256", "split_manifest_sha256")):
            raise ValueError(f"{model_name} was not trained on the frozen final dataset/split")
    if model_name != "Logistic Regression":
        config = read_json(path / "config.json")
        if ({int(k): v for k, v in config["id2label"].items()} != internal
                or config["label2id"] != {v: k for k, v in internal.items()}):
            raise ValueError(f"{model_name} config and project label mapping disagree")
        expected_type = "bert" if model_name == "BERT" else "distilbert"
        tokenizer_config = read_json(path / "tokenizer_config.json")
        if config["model_type"] != expected_type or config["max_position_embeddings"] < TRANSFORMER_MAX_LENGTH:
            raise ValueError("Unexpected model architecture or input limit")
        if tokenizer_config["model_max_length"] != TRANSFORMER_MAX_LENGTH:
            raise ValueError("Tokenizer input limit differs from the trained 512-token configuration")
    return {"model_name": model_name, "model_path": str(path),
            "tokenizer_path": str(path) if model_name != "Logistic Regression" else None,
            "project_labels": dict(sorted(projects.items())), "internal_labels": internal,
            "internal_to_project": to_project,
            "max_length": TRANSFORMER_MAX_LENGTH if model_name != "Logistic Regression" else None}


@lru_cache(maxsize=3)
def _load_classifier(model_name, signature):
    metadata = get_model_metadata(model_name)
    path = MODEL_PATHS[model_name]
    if model_name == "Logistic Regression":
        import joblib
        model = joblib.load(path / "pipeline.joblib")
        if set(map(int, model.classes_)) != set(metadata["internal_labels"]):
            raise ValueError("Baseline classes disagree with the final mapping")
        resources = (None, model)
        weights_path = path / "pipeline.joblib"
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
        if model.config.id2label != metadata["internal_labels"]:
            raise ValueError("Loaded classifier differs from its saved labels")
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        resources = (tokenizer, model)
        weights_path = path / "model.safetensors"
    weights_hash = sha256(weights_path)
    registry_path = ROOT / "results/final_model_comparison/model_registry.json"
    if registry_path.is_file() and read_json(registry_path)["models"][model_name]["weights_sha256"] != weights_hash:
        raise ValueError(f"{model_name} weights do not match the final evaluated model registry")
    return resources, metadata, weights_hash


def _resources(model_name):
    signature = tuple(item for item in artifact_fingerprint() if item[0] in (model_name, "registry"))
    return _load_classifier(model_name, signature)


def load_classifier(model_name="DistilBERT"):
    """Compatibility helper; loaded once per model/artifact version."""
    with _INFERENCE_LOCK:
        return _resources(model_name)[0]


def _normalize_text(text):
    return " ".join(str(text).split()) if text is not None else ""


def create_encoded_chunks(text, tokenizer, max_tokens=TRANSFORMER_MAX_LENGTH,
                          overlap_tokens=CHUNK_OVERLAP_TOKENS):
    """Cover every token, reserving room for special tokens.

    Inference uses token IDs directly to avoid decode/re-encode token loss.
    This does not mutate the shared tokenizer's maximum length.
    """
    capacity = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity <= 0 or not 0 <= overlap_tokens < capacity:
        raise ValueError("Chunk size must leave space for content and exceed the overlap")
    clean_text = _normalize_text(text)
    if not clean_text:
        return []
    direct = tokenizer(clean_text, truncation=False, return_special_tokens_mask=True,
                       return_token_type_ids=False, verbose=False)
    if len(direct["input_ids"]) <= max_tokens:
        return [direct] if any(value == 0 for value in direct["special_tokens_mask"]) else []
    encoded = tokenizer(clean_text, truncation=True, max_length=max_tokens,
                        stride=overlap_tokens, return_overflowing_tokens=True,
                        return_special_tokens_mask=True, return_token_type_ids=False,
                        padding=False)
    return [{key: encoded[key][i] for key in ("input_ids", "attention_mask", "special_tokens_mask")}
            for i in range(len(encoded["input_ids"]))
            if any(value == 0 for value in encoded["special_tokens_mask"][i])]


def create_token_chunks(text, tokenizer, max_tokens=TRANSFORMER_MAX_LENGTH,
                        overlap_tokens=CHUNK_OVERLAP_TOKENS):
    """Return content tokens without the tokenizer-added special tokens."""
    return [[token for token, special in zip(chunk["input_ids"], chunk["special_tokens_mask"]) if not special]
            for chunk in create_encoded_chunks(text, tokenizer, max_tokens, overlap_tokens)]


def create_chunks(text, tokenizer, max_tokens=TRANSFORMER_MAX_LENGTH,
                  overlap_tokens=CHUNK_OVERLAP_TOKENS):
    """Retain the decoded-chunk helper for callers displaying text chunks."""
    return [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for ids in create_token_chunks(text, tokenizer, max_tokens, overlap_tokens)]


def aggregate_chunk_probabilities(chunk_probabilities):
    if not chunk_probabilities:
        raise ValueError("At least one chunk is required for aggregation")
    probabilities = np.mean(np.stack(chunk_probabilities), axis=0)
    return probabilities / probabilities.sum()



def predict_model(text, model_name="DistilBERT"):
    """One classifier result; classes are mapped through saved metadata."""
    if model_name not in MODEL_PATHS:
        raise ValueError("Unknown final classifier")
    if text is not None and len(str(text)) > MAX_INPUT_CHARACTERS:
        raise ValueError(f"Please limit each analysis to {MAX_INPUT_CHARACTERS:,} characters")
    clean_text = _normalize_text(text)
    if not clean_text:
        raise ValueError("Input text cannot be empty")
    with _INFERENCE_LOCK:
        (tokenizer, model), metadata, weights_hash = _resources(model_name)
        started = perf_counter()
        if model_name == "Logistic Regression":
            raw = model.predict_proba([clean_text])[0]
            averaged = np.zeros(6)
            for class_id, probability in zip(model.classes_, raw):
                averaged[int(class_id)] = probability
            chunk_count = 1
        else:
            import torch
            chunks = create_encoded_chunks(clean_text, tokenizer)
            if not chunks:
                raise ValueError("Input text contains no usable tokens")
            values = []
            with torch.inference_mode():
                for chunk in chunks:
                    encoded = {key: torch.tensor([chunk[key]], dtype=torch.long, device=model.device)
                               for key in ("input_ids", "attention_mask")}
                    values.append(model(**encoded).logits.softmax(dim=-1)[0].cpu().numpy())
            averaged = aggregate_chunk_probabilities(values)
            chunk_count = len(chunks)
        elapsed = perf_counter() - started
    index = int(np.argmax(averaged))
    by_name = {name: float(averaged[i]) for i, name in metadata["internal_labels"].items()}
    return {"model_name": model_name, "predicted_class": index,
            "predicted_project_label": metadata["internal_to_project"][index],
            "predicted_label": metadata["internal_labels"][index], "confidence": float(averaged[index]),
            "probability_by_label": {name: by_name[name] for name in metadata["project_labels"].values()},
            "number_of_chunks": chunk_count, "model_path": metadata["model_path"],
            "tokenizer_path": metadata["tokenizer_path"], "weights_sha256": weights_hash,
            "inference_seconds": elapsed}


def predict_text(text):
    """BERT, corrected DistilBERT and TF-IDF/LR on the same input."""
    if not _normalize_text(text):
        raise ValueError("Input text cannot be empty")
    return {name: predict_model(text, name) for name in MODEL_NAMES}


def model_agreement(results):
    from collections import Counter
    counts = Counter(result["predicted_label"] for result in results.values())
    if not counts:
        raise ValueError("No classifier predictions to compare")
    label, count = counts.most_common(1)[0]
    if count == len(results):
        message = f"All {count} models agree: {label}"
    elif count > 1:
        message = f"{count} of {len(results)} models agree: {label}"
    else:
        message = "Models produced different predictions"
    return {"message": message, "agreeing_models": count, "total_models": len(results),
            "predictions": {name: value["predicted_label"] for name, value in results.items()}}


def predict_sentiment(text):
    from sentiment import analyze_vader_sentiment
    return analyze_vader_sentiment(text)


if __name__ == "__main__":
    print("Fake News & Content Classification System")
    while True:
        text = input("\nEnter text (or EXIT):\n> ")
        if text.strip().lower() == "exit":
            break
        try:
            print(json.dumps(predict_text(text), indent=2))
            print("VADER sentiment:", json.dumps(predict_sentiment(text)))
        except (ValueError, OSError) as error:
            print("Analysis could not be completed:", error)
