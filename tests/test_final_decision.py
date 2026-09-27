"""Decision features, leakage boundaries, probability metrics and saved runtime parity."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import final_decision as decision
from run_final_decision_experiment import OUT, group_folds, metrics, verify_protected


class DecisionUnitTests(unittest.TestCase):
    def test_exact_feature_order(self):
        p = np.array([[.1, .2, .3, .1, .2, .1]])
        sentiment = np.array([[.2, .5, .3, -.1]])
        np.testing.assert_allclose(decision.decision_features(p, sentiment), np.column_stack((p, sentiment)))
        self.assertEqual(decision.decision_features(p).shape, (1, 6))
        self.assertEqual(decision.decision_features(p, sentiment).shape, (1, 10))

    def test_invalid_features_fail_closed(self):
        p = np.full((1, 6), 1/6)
        for bad in [np.zeros((1, 6)), np.full((1, 6), np.nan), np.ones((1, 5)),
                    np.array([[1.1, -.1, 0, 0, 0, 0]]), np.empty((0, 6))]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decision.decision_features(bad)
        for bad in [[[0, 1, 0]], [[0, 1, 0, 1.1]], [[0, .5, 0, 0]], [[0, 1, 0, float('inf')]]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decision.decision_features(p, bad)
        self.assertEqual(decision.decision_features(p, [[0, 0, 0, 0]]).shape, (1, 10))

    def test_probability_only_never_uses_sentiment(self):
        p = np.full((1, 6), 1/6)
        np.testing.assert_allclose(decision.apply_decision(p, None, "classifier_alone"), p)
        with self.assertRaises(ValueError):
            decision.apply_decision(p, None, "unknown")

    def test_metrics_have_known_brier_and_log_loss(self):
        y = np.arange(6)
        perfect = metrics(y, np.eye(6))
        self.assertEqual(perfect["accuracy"], 1)
        self.assertEqual(perfect["macro_f1"], 1)
        self.assertEqual(perfect["multiclass_brier"], 0)
        uniform = metrics(y, np.full((6, 6), 1/6))
        self.assertAlmostEqual(uniform["log_loss"], np.log(6))
        self.assertAlmostEqual(uniform["multiclass_brier"], 5/6)

    def test_vader_same_text_normalization_and_negation(self):
        plain = decision.vader_features(["This is excellent!"])
        spaced = decision.vader_features(["  This  is\nexcellent! "])
        np.testing.assert_array_equal(plain, spaced)
        negated = decision.vader_features(["This is not excellent!"])
        self.assertGreater(plain[0, 3], negated[0, 3])
        for value in ("", "  ", None):
            with self.assertRaises(ValueError): decision.vader_features([value])

    def test_group_folds_keep_related_rows_together(self):
        y = np.tile(np.arange(6), 30)
        groups = np.array([f"{i//12}:{i%6}" for i in range(len(y))])
        folds = group_folds(y, groups, 5, 42)
        for train, held in folds:
            self.assertFalse(set(groups[train]) & set(groups[held]))


@unittest.skipUnless((OUT / "fusion_completed.json").is_file(), "Run versioned fusion experiment first")
class SavedDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(OUT / "fusion_features.csv", keep_default_na=False)
        cls.metadata = json.loads((decision.ARTIFACT_DIR / "deployment.json").read_text())

    def test_protected_inputs_and_fixed_selection(self):
        self.assertGreater(verify_protected(), 14)
        self.assertEqual(self.metadata["selected_classifier"], "DistilBERT")
        self.assertFalse(self.metadata["test_used_for_selection"])
        self.assertEqual(self.metadata["label_mapping"], decision.MAPPING)

    def test_saved_folds_are_validation_only_and_nested_without_group_overlap(self):
        folds = pd.read_csv(OUT / "development_folds.csv", keep_default_na=False)
        dev = set(self.frame.loc[self.frame.split == "validation", "dataset_row"])
        self.assertTrue(set(folds.dataset_row) <= dev)
        outer = {}
        for (outer_id, inner_id), frame in folds.groupby(["outer_fold", "inner_fold"]):
            train = frame[frame.role == "fit"]
            held = frame[frame.role == "heldout"]
            self.assertFalse(set(train.group_id) & set(held.group_id))
            self.assertFalse(set(train.dataset_row) & set(held.dataset_row))
            if inner_id == -1: outer[outer_id] = set(train.dataset_row)
        for (outer_id, inner_id), frame in folds.groupby(["outer_fold", "inner_fold"]):
            if outer_id >= 0 and inner_id >= 0:
                self.assertTrue(set(frame.dataset_row) <= outer[outer_id])
        held = folds[(folds.inner_fold == -1) & (folds.role == "heldout")]
        self.assertEqual(set(held.dataset_row), dev)
        self.assertEqual(len(held), len(dev))

    def test_serialized_decisions_match_all_saved_test_probabilities(self):
        test = self.frame[self.frame.split == "test"]
        p = test[list(decision.PROBABILITY_FEATURES)].to_numpy()
        s = test[list(decision.SENTIMENT_FEATURES)].to_numpy()
        for system in decision.SYSTEMS:
            actual = decision.apply_decision(p, s, system)
            saved = pd.read_csv(OUT / "fusion" / system / "test_predictions.csv")
            self.assertEqual(test.dataset_row.tolist(), saved.dataset_row.tolist())
            np.testing.assert_allclose(actual, saved[list(decision.PROBABILITY_FEATURES)].to_numpy(), atol=1e-12)
        with self.assertRaises(ValueError): decision.apply_decision(p, None, "vader_fusion")
        a = decision.apply_decision(p, s, "probability_only")
        b = decision.apply_decision(p, None, "probability_only")
        np.testing.assert_array_equal(a, b)

    def test_final_scalers_fit_only_validation_features(self):
        import joblib
        dev = self.frame[self.frame.split == "validation"]
        for system in decision.SYSTEMS[1:]:
            entry = self.metadata["artifacts"][system]
            layer = joblib.load(decision.ARTIFACT_DIR / entry["filename"])
            scaler = layer.steps[0][1]
            self.assertEqual(int(scaler.n_samples_seen_), 1269)
            np.testing.assert_allclose(scaler.mean_, dev[entry["feature_names"]].to_numpy().mean(axis=0), atol=1e-12)
            self.assertEqual(layer.n_features_in_, 10 if system == "vader_fusion" else 6)

    def test_all_partitions_and_train_metrics_are_complete(self):
        frame = pd.read_csv(OUT / "classifier_comparison.csv")
        self.assertEqual(len(frame), 9)
        for row in frame.itertuples():
            self.assertEqual(row.records, 5921 if row.split == "train" else 1269)
        for name in ("bert", "distilbert", "logistic_regression"):
            saved = pd.read_csv(OUT / "classifiers" / name / "probabilities.csv")
            self.assertEqual(saved.dataset_row.tolist(), list(range(8459)))
            for split in ("train", "validation", "test"):
                part = saved[saved.split == split]
                actual = metrics(part.internal_label.to_numpy(), part[list(decision.PROBABILITY_FEATURES)].to_numpy())
                report = json.loads((OUT / "classifiers" / name / f"{split}_metrics.json").read_text())
                for key in ("accuracy", "macro_f1", "weighted_f1", "log_loss", "multiclass_brier"):
                    self.assertAlmostEqual(actual[key], report[key], places=12)

    def test_decision_artifact_hash_failure_is_rejected(self):
        bad = json.loads(json.dumps(self.metadata))
        bad["artifacts"]["vader_fusion"]["sha256"] = "invalid"
        with patch.object(decision, "read_json", return_value=bad):
            decision._load_layer.cache_clear()
            with self.assertRaises(ValueError):
                decision.apply_decision(np.full((1, 6), 1/6), [[0, 1, 0, 0]], "vader_fusion")
        decision._load_layer.cache_clear()

    def test_feature_text_hash_and_vader_reproduction(self):
        import hashlib
        import csv
        with (ROOT / "data/processed/final_six_category_dataset.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        ids = [0, 6, 42, 1000, 8458]
        for i in ids:
            self.assertEqual(self.frame.iloc[i].text_sha256, hashlib.sha256(rows[i]["text"].encode()).hexdigest())
        np.testing.assert_allclose(decision.vader_features([rows[i]["text"] for i in ids]),
                                   self.frame.iloc[ids][list(decision.SENTIMENT_FEATURES)].to_numpy(dtype=float))

    def test_final_runtime_returns_one_prediction_for_each_variant(self):
        text = "This article is brought to you by TravelPro. Book your next vacation with our exclusive summer deals."
        from predict import predict_model
        base = predict_model(text, self.metadata["selected_classifier"])
        p = [[base["probability_by_label"][label] for label in decision.CATEGORIES.values()]]
        for system in decision.SYSTEMS:
            result = decision.predict_final(text, system)
            expected = decision.apply_decision(p, decision.vader_features([text]), system)[0]
            self.assertEqual(result["predicted_project_label"], int(expected.argmax())+1)
            self.assertEqual(result["predicted_label"], decision.CATEGORIES[result["predicted_project_label"]])
            self.assertAlmostEqual(result["confidence"], float(expected.max()))
            self.assertEqual(len(result["probability_by_label"]), 6)
            self.assertNotIn("classification", result)
        self.assertEqual(decision.predict_final(text)["system"], self.metadata["recommended_system"])

    def test_sentiment_failure_and_invalid_inputs_do_not_silently_fallback(self):
        with patch.object(decision, "vader_features", side_effect=RuntimeError("unavailable")):
            with self.assertRaises(RuntimeError): decision.predict_final("Example text.", "vader_fusion")
        for text in (None, "", "   ", "x"*50001):
            with self.subTest(text_length=len(text) if text else 0), self.assertRaises(ValueError):
                decision.predict_final(text, "vader_fusion")

    def test_changed_classifier_or_mapping_is_rejected(self):
        bad = dict(self.metadata, label_mapping={})
        with patch.object(decision, "read_json", return_value=bad):
            with self.assertRaises(ValueError): decision.predict_final("Example")
        import predict
        with patch.object(predict, "predict_model", return_value={"weights_sha256": "wrong"}):
            with self.assertRaises(ValueError): decision.predict_final("Example")


if __name__ == "__main__":
    unittest.main()
