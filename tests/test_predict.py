"""Offline checks for the production classifier, token coverage and VADER."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import predict
from sentiment import analyze_vader_sentiment

EXPECTED_LABELS = [
    "Native Advertising", "News Satire", "Propaganda",
    "Manipulation", "News Parody", "Fabrication",
]


class PredictTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer, cls.model = predict.load_classifier()

    def test_empty_input_raises_value_error(self):
        for text in ("", "   ", None):
            with self.subTest(text=text), self.assertRaises(ValueError):
                predict.predict_model(text)

    def test_short_input_returns_final_model_result(self):
        result = predict.predict_model("Sponsored: Discover our new running shoes and shop the collection today.")
        self.assertEqual(list(result["probability_by_label"]), EXPECTED_LABELS)
        self.assertEqual(result["predicted_label"], EXPECTED_LABELS[result["predicted_project_label"] - 1])
        self.assertAlmostEqual(result["confidence"], max(result["probability_by_label"].values()), places=6)
        self.assertAlmostEqual(sum(result["probability_by_label"].values()), 1.0, places=5)
        self.assertEqual(result["number_of_chunks"], 1)
        self.assertEqual(Path(result["model_path"]), ROOT / "models/distilbert_six_category_native_ads_corrected")
        self.assertTrue(all(0 <= value <= 1 for value in result["probability_by_label"].values()))
        self.assertNotIn("BERT", result)
        self.assertNotIn("Logistic Regression", result)
        self.assertNotIn("sentiment", result)

    def test_saved_mapping_and_token_limit_match_training(self):
        metadata = predict.get_model_metadata()
        self.assertEqual(list(metadata["project_labels"].values()), EXPECTED_LABELS)
        self.assertEqual(metadata["internal_labels"], self.model.config.id2label)
        plan = json.loads((ROOT / "results/distilbert_six_category_native_ads_corrected/training_plan.json").read_text())
        self.assertEqual(metadata["max_length"], plan["hyperparameters"]["max_length"])
        self.assertEqual(predict.TRANSFORMER_MAX_LENGTH, 512)

    def test_single_chunk_matches_direct_saved_model_inference(self):
        import torch
        text = "This article is brought to you by TravelPro. Book your next vacation with our exclusive summer deals."
        encoded = self.tokenizer(text, return_tensors="pt", return_token_type_ids=False)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            expected = self.model(**encoded).logits.softmax(dim=-1)[0].cpu().numpy()
        actual = predict.predict_model(text)
        for index, category in self.model.config.id2label.items():
            self.assertAlmostEqual(actual["probability_by_label"][category], float(expected[index]), places=6)

    def test_exact_512_token_boundary(self):
        capacity = 512 - self.tokenizer.num_special_tokens_to_add(pair=False)
        full = predict.create_encoded_chunks("word " * capacity, self.tokenizer)
        over = predict.create_encoded_chunks("word " * (capacity + 1), self.tokenizer)
        self.assertEqual(len(full), 1)
        self.assertEqual(len(full[0]["input_ids"]), 512)
        self.assertEqual(len(over), 2)
        self.assertTrue(all(len(chunk["input_ids"]) <= 512 for chunk in over))

    def test_long_text_preserves_every_token_and_overlap(self):
        text = "Uncharacteristically, tokenization preserves multilingual café details. " * 100
        before_limit = self.tokenizer.model_max_length
        expected = self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        chunks = predict.create_token_chunks(text, self.tokenizer)
        self.assertGreater(len(chunks), 1)
        recovered = list(chunks[0])
        overlap = predict.CHUNK_OVERLAP_TOKENS
        for left, right in zip(chunks, chunks[1:]):
            self.assertEqual(left[-overlap:], right[:overlap])
            recovered.extend(right[overlap:])
        self.assertEqual(recovered, expected)
        self.assertEqual(self.tokenizer.model_max_length, before_limit)

    def test_long_prediction_and_aggregation(self):
        text = "A local council approved a public transport plan. " * 65
        result = predict.predict_model(text)
        self.assertGreater(result["number_of_chunks"], 1)
        self.assertAlmostEqual(sum(result["probability_by_label"].values()), 1.0, places=5)
        averaged = predict.aggregate_chunk_probabilities([np.array([.8, .2]), np.array([.2, .8])])
        np.testing.assert_allclose(averaged, [.5, .5])

    def test_decoded_chunk_helper_remains_available(self):
        chunks = predict.create_chunks("word " * 600, self.tokenizer)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_invalid_chunk_settings_and_oversized_input_are_rejected(self):
        for size, overlap in ((2, 0), (512, 510), (512, -1)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                predict.create_token_chunks("text", self.tokenizer, size, overlap)
        with self.assertRaises(ValueError):
            predict.predict_model("x" * (predict.MAX_INPUT_CHARACTERS + 1))
        with self.assertRaises(ValueError):
            predict.aggregate_chunk_probabilities([])

    def test_model_is_cached_and_in_evaluation_mode(self):
        tokenizer, model = predict.load_classifier()
        self.assertIs(model, self.model)
        self.assertIs(tokenizer, self.tokenizer)
        self.assertFalse(model.training)

    def test_missing_artifacts_fail_without_fallback(self):
        with patch.dict(predict.MODEL_PATHS, {"DistilBERT": ROOT / "missing_final_model_for_test"}):
            with self.assertRaisesRegex(FileNotFoundError, "Final local DistilBERT model is incomplete"):
                predict.get_model_metadata()

    def test_all_three_final_models_and_saved_split(self):
        result = predict.predict_text("This article is brought to you by TravelPro. Book your next vacation with our exclusive summer deals.")
        self.assertEqual(list(result), ["BERT", "DistilBERT", "Logistic Regression"])
        for name, value in result.items():
            metadata = predict.get_model_metadata(name)
            self.assertEqual(list(value["probability_by_label"]), EXPECTED_LABELS)
            self.assertEqual(value["predicted_label"], EXPECTED_LABELS[value["predicted_project_label"] - 1])
            self.assertAlmostEqual(sum(value["probability_by_label"].values()), 1.0, places=5)
            self.assertEqual(value["model_path"], metadata["model_path"])
            self.assertEqual(len(value["weights_sha256"]), 64)
        self.assertEqual(result["DistilBERT"]["predicted_project_label"], 1)
        self.assertAlmostEqual(result["DistilBERT"]["confidence"], 0.6786684393882751, places=5)

    def test_bert_complete_token_coverage(self):
        tokenizer, _ = predict.load_classifier("BERT")
        text = "Uncharacteristically, technical tokenization preserves every subword. " * 100
        expected = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        chunks = predict.create_token_chunks(text, tokenizer)
        self.assertGreater(len(chunks), 1)
        recovered = list(chunks[0])
        for left, right in zip(chunks, chunks[1:]):
            self.assertEqual(left[-16:], right[:16])
            recovered.extend(right[16:])
        self.assertEqual(recovered, expected)
        self.assertTrue(all(len(c) <= 510 for c in chunks))

    def test_bert_export_is_the_complete_best_checkpoint(self):
        from final_project import read_json, sha256
        best = read_json(ROOT / "results/bert_six_category_final/best_checkpoint.json")
        actual = sha256(ROOT / "models/bert_six_category_final/model.safetensors")
        registry = read_json(ROOT / "results/final_model_comparison/model_registry.json")
        self.assertEqual(actual, registry["models"]["BERT"]["weights_sha256"])
        checkpoint = ROOT / "models/bert_six_category_final/checkpoints" / Path(best["checkpoint"].replace("\\", "/")).name / "model.safetensors"
        if checkpoint.is_file():
            self.assertEqual(actual, sha256(checkpoint),
                             "Export must include all best-checkpoint parameters, including LayerNorm")

    def test_agreement_is_descriptive_and_handles_all_outcomes(self):
        for labels, count in [(EXPECTED_LABELS[:3], 1), (["Propaganda"] * 3, 3),
                              (["Propaganda", "Propaganda", "News Satire"], 2)]:
            values = {name: {"predicted_label": label} for name, label in zip(predict.MODEL_NAMES, labels)}
            self.assertEqual(predict.model_agreement(values)["agreeing_models"], count)

    def test_vader_positive_text(self):
        result = analyze_vader_sentiment("I am absolutely thrilled and delighted with this amazing success!")
        self.assertEqual(result["sentiment_label"], "Positive")
        self.assertGreater(result["compound"], 0.0)
        self.assertGreater(result["positive"], result["negative"])

    def test_vader_negative_text(self):
        result = analyze_vader_sentiment("This is a terrible, disappointing, and deeply upsetting situation.")
        self.assertEqual(result["sentiment_label"], "Negative")
        self.assertLess(result["compound"], 0.0)
        self.assertGreater(result["negative"], result["positive"])

    def test_vader_neutral_text(self):
        result = analyze_vader_sentiment("The report states the meeting will begin at 9 a.m. and include six attendees.")
        self.assertEqual(result["sentiment_label"], "Neutral")
        self.assertAlmostEqual(result["compound"], 0.0, delta=0.2)

    def test_vader_empty_input(self):
        result = analyze_vader_sentiment("   ")
        self.assertEqual(result["sentiment_label"], "Neutral")
        self.assertEqual(result["positive"], 0.0)
        self.assertEqual(result["neutral"], 1.0)
        self.assertEqual(result["negative"], 0.0)
        self.assertEqual(result["compound"], 0.0)

    def test_vader_long_text(self):
        long_text = " ".join([
            "The local community meeting was calm and informative.",
            "Officials explained the project plan and public response.",
            "People listened carefully and asked thoughtful questions.",
            "The atmosphere remained steady and constructive throughout the session.",
            "No severe problems were reported, and the discussion continued without incident."
        ] * 15)
        result = analyze_vader_sentiment(long_text)
        self.assertIn("sentiment_label", result)
        self.assertIn("positive", result)
        self.assertIn("neutral", result)
        self.assertIn("negative", result)
        self.assertIn("compound", result)

    def test_vader_output_contains_expected_fields(self):
        result = analyze_vader_sentiment("This policy change surprised many people and sparked strong debate.")
        expected_fields = {"sentiment_label", "positive", "neutral", "negative", "compound"}
        self.assertSetEqual(set(result.keys()), expected_fields)


if __name__ == "__main__":
    unittest.main()
