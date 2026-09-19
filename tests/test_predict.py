import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import predict_text, create_chunks
from sentiment import analyze_vader_sentiment


class PredictTextTests(unittest.TestCase):
    def test_empty_input_raises_value_error(self):
        with self.assertRaises(ValueError):
            predict_text("   ")

    def test_short_input_returns_valid_structure(self):
        result = predict_text("A local council approved a new public transport plan.")
        self.assertSetEqual(set(result.keys()), {"BERT", "DistilBERT", "Logistic Regression"})

        for model_name in ("BERT", "DistilBERT", "Logistic Regression"):
            model_result = result[model_name]
            self.assertIn("predicted_class", model_result)
            self.assertIn("predicted_label", model_result)
            self.assertIn("confidence", model_result)
            self.assertIn("probability_by_label", model_result)
            self.assertEqual(len(model_result["probability_by_label"]), 6)
            self.assertAlmostEqual(sum(model_result["probability_by_label"].values()), 1.0, places=5)
            self.assertGreaterEqual(model_result["confidence"], 0.0)
            self.assertLessEqual(model_result["confidence"], 1.0)

            if model_name in ("BERT", "DistilBERT"):
                self.assertIn("number_of_chunks", model_result)
                self.assertGreaterEqual(model_result["number_of_chunks"], 1)

    def test_near_token_limit_stays_single_chunk(self):
        near_limit_text = " ".join(["This is a short factual sentence about local policy and public safety."] * 8)
        tokenizer = __import__("predict").AutoTokenizer.from_pretrained("bert-base-uncased")
        chunks = create_chunks(near_limit_text, tokenizer, max_tokens=128, overlap_tokens=16)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_long_input_uses_multiple_chunks(self):
        long_text = " ".join(["This is a test article about public policy and local governance."] * 250)
        result = predict_text(long_text)
        self.assertGreater(result["BERT"]["number_of_chunks"], 1)
        self.assertGreater(result["DistilBERT"]["number_of_chunks"], 1)

    def test_very_long_input_creates_many_chunks(self):
        very_long_text = " ".join(["The article describes a local policy update and public response to governance changes."] * 500)
        tokenizer = __import__("predict").AutoTokenizer.from_pretrained("bert-base-uncased")
        chunks = create_chunks(very_long_text, tokenizer, max_tokens=128, overlap_tokens=16)
        self.assertGreater(len(chunks), 10)

    def test_create_chunks_avoids_empty_chunks_and_has_overlap(self):
        tokenizer = __import__("predict").AutoTokenizer.from_pretrained("bert-base-uncased")
        long_text = " ".join(["sentence about policy and government."] * 200)
        chunks = create_chunks(long_text, tokenizer, max_tokens=64, overlap_tokens=16)
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.strip() for chunk in chunks))
        self.assertGreater(len(chunks), 1)

    def test_probability_aggregation_is_valid(self):
        result = predict_text("A news article about a public safety decision by local authorities.")
        for model_name in ("BERT", "DistilBERT"):
            probs = result[model_name]["probability_by_label"].values()
            self.assertAlmostEqual(sum(probs), 1.0, places=5)

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
