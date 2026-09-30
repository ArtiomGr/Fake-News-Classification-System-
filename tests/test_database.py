"""Tests for anonymous analysis-history persistence."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from database import get_recent_analyses, save_analysis


class DatabaseTests(unittest.TestCase):
    def test_save_and_read_analysis_without_storing_text(self):
        result = {
            "system": "classifier_alone",
            "classifier": "DistilBERT",
            "predicted_label": "Native Advertising",
            "confidence": 0.9,
            "number_of_chunks": 1,
            "inference_seconds": 0.12,
            "sentiment": {"compound": 0.8},
            "sentiment_label": "Positive",
            "compound": 0.8,
            "positive": 0.8,
            "neutral": 0.2,
            "negative": 0.0,
            "vader_compound": 0.8,
        }
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "history.json"
            save_analysis("private article text", result, database_path)

            rows = get_recent_analyses(database_path=database_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["predicted_label"], "Native Advertising")
            self.assertEqual(rows[0]["text_length"], len("private article text"))
            self.assertNotIn("private article text", str(rows[0]))

            saved = json.loads(database_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved), 1)
            self.assertNotIn("text", saved[0])
            self.assertEqual(set(saved[0]), {
                "id", "analyzed_at", "text_sha256", "text_length", "system", "classifier",
                "predicted_label", "confidence", "number_of_chunks", "inference_seconds",
            })


if __name__ == "__main__":
    unittest.main()
