"""Frozen six-category classifier plus an optional learned decision layer.

No fitting, downloads, or filesystem writes occur on import or prediction.
The Streamlit application deliberately does not import this module yet.
"""
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from final_project import CATEGORIES, MAPPING, ROOT, read_json, sha256

ARTIFACT_DIR = ROOT / "models/final_decision_v1"
SYSTEMS = ("classifier_alone", "probability_only", "vader_fusion")
PROBABILITY_FEATURES = tuple(f"p_{i}_{name.lower().replace(' ', '_')}" for i, name in CATEGORIES.items())
SENTIMENT_FEATURES = ("vader_neg", "vader_neu", "vader_pos", "vader_compound")


def probability_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not len(values):
        raise ValueError("Expected a nonempty N x 6 probability matrix")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Each probability vector must sum to one")
    return values / values.sum(axis=1, keepdims=True)


def decision_features(probabilities, sentiment=None):
    probabilities = probability_matrix(probabilities)
    if sentiment is None:
        return probabilities
    sentiment = np.asarray(sentiment, dtype=np.float64)
    if sentiment.shape != (len(probabilities), 4) or not np.isfinite(sentiment).all():
        raise ValueError("Expected finite N x 4 VADER features in neg, neu, pos, compound order")
    if ((sentiment[:, :3] < 0) | (sentiment[:, :3] > 1)).any() or (np.abs(sentiment[:, 3]) > 1).any():
        raise ValueError("VADER features out of range")
    # VADER may return all-zero proportions for punctuation-only input.
    sums = sentiment[:, :3].sum(axis=1)
    if not np.all(np.isclose(sums, 1, atol=.002) | np.isclose(sums, 0, atol=1e-8)):
        raise ValueError("VADER proportions must sum to one (or zero for no lexical content)")
    return np.column_stack((probabilities, sentiment))


@lru_cache(maxsize=1)
def _analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def vader_features(texts):
    """Same whitespace normalization as classifier inference; retain case/punctuation."""
    values = []
    for text in texts:
        clean = " ".join(str(text).split()) if text is not None else ""
        if not clean:
            raise ValueError("Input text cannot be empty")
        scores = _analyzer().polarity_scores(clean)
        values.append([scores[key] for key in ("neg", "neu", "pos", "compound")])
    return np.asarray(values, dtype=np.float64).reshape(-1, 4)


def make_estimator(regularization):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # Scaling is fitted within each development fold, never on test data.
    # lbfgs uses multinomial loss for this six-class task, with default L2.
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=float(regularization), solver="lbfgs", max_iter=3000, tol=1e-8,
        class_weight=None, random_state=42))


@lru_cache(maxsize=8)
def _load_layer(directory, system, metadata_hash, artifact_hash):
    import joblib
    directory = Path(directory)
    metadata = read_json(directory / "deployment.json")
    if metadata["label_mapping"] != MAPPING:
        raise ValueError("Decision layer label mapping mismatch")
    expected = list(PROBABILITY_FEATURES + (SENTIMENT_FEATURES if system == "vader_fusion" else ()))
    entry = metadata["artifacts"][system]
    if entry["feature_names"] != expected or entry["sha256"] != artifact_hash:
        raise ValueError("Decision layer feature order or artifact hash mismatch")
    filename = directory / entry["filename"]
    if filename.resolve().parent != directory.resolve():
        raise ValueError("Decision artifact must be inside its declared directory")
    model = joblib.load(filename)
    if model.classes_.tolist() != list(range(6)) or model.n_features_in_ != len(expected):
        raise ValueError("Decision layer classes or feature count mismatch")
    return model


def apply_decision(probabilities, sentiment, system, artifact_dir=ARTIFACT_DIR):
    if system not in SYSTEMS:
        raise ValueError("Unknown final decision system")
    probabilities = probability_matrix(probabilities)
    if system == "classifier_alone":
        return probabilities
    directory = Path(artifact_dir)
    metadata = read_json(directory / "deployment.json")
    entry = metadata["artifacts"][system]
    filename = directory / entry["filename"]
    if filename.resolve().parent != directory.resolve():
        raise ValueError("Invalid decision artifact path")
    layer = _load_layer(str(directory), system, sha256(directory / "deployment.json"), sha256(filename))
    if system == "vader_fusion" and sentiment is None:
        raise ValueError("VADER fusion requires sentiment features; no silent fallback")
    features = decision_features(probabilities, sentiment if system == "vader_fusion" else None)
    return probability_matrix(layer.predict_proba(features))


def predict_final(text, system=None, artifact_dir=ARTIFACT_DIR):
    """Return exactly one selected system's category, confidence and six probabilities.

    An explicit system selects an experimental variant. The default follows the
    development-only deployment recommendation recorded before test evaluation.
    """
    from predict import predict_model
    started = perf_counter()
    metadata = read_json(Path(artifact_dir) / "deployment.json")
    if metadata["label_mapping"] != MAPPING:
        raise ValueError("Final category mapping mismatch")
    system = metadata["recommended_system"] if system is None else system
    if system not in SYSTEMS:
        raise ValueError("Unknown final decision system")
    base = predict_model(text, metadata["selected_classifier"])
    if base["weights_sha256"] != metadata["selected_classifier_weights_sha256"]:
        raise ValueError("Classifier weights differ from decision-layer development")
    probabilities = [[base["probability_by_label"][label] for label in CATEGORIES.values()]]
    sentiment = vader_features([text]) if system == "vader_fusion" else None
    final = apply_decision(probabilities, sentiment, system, artifact_dir)[0]
    index = int(final.argmax())
    return {"system": system, "classifier": metadata["selected_classifier"],
            "predicted_class": index, "predicted_project_label": index + 1,
            "predicted_label": CATEGORIES[index + 1], "confidence": float(final[index]),
            "probability_by_label": dict(zip(CATEGORIES.values(), map(float, final))),
            "confidence_interpretation": "model probability; not independently calibrated factual certainty",
            "number_of_chunks": base["number_of_chunks"],
            "inference_seconds": perf_counter() - started}
