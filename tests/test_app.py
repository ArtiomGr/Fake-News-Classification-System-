"""Final dashboard interaction and exact TravelPro regression checks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import predict

LABELS = ["Native Advertising", "News Satire", "Propaganda", "Manipulation", "News Parody", "Fabrication"]
TRAVELPRO = "This article is brought to you by TravelPro. Book your next vacation with our exclusive summer deals."


class FinalDashboardTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=180).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_academic_identity_and_final_categories(self):
        app = self.app()
        sidebar = "\n".join(item.value for item in app.sidebar.caption)
        for index, label in enumerate(LABELS, 1):
            self.assertIn(f"{index}. {label}", sidebar)
        rendered = "\n".join(item.value for item in [*app.title, *app.markdown]) + sidebar
        self.assertIn("Fake News & Content Classification System", rendered)
        self.assertNotIn("TruthLens", rendered)
        for label in ["False Connection", "Imposter Content", "Manipulated Content", "Misleading Content"]:
            self.assertNotIn(label, rendered)

    def test_empty_input_is_explained(self):
        app = self.app()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.warning[0].value, "Please enter text before running the analysis.")
        self.assertEqual(len(app.metric), 0)

    def test_three_models_travelpro_parity_sentiment_and_clear(self):
        app = self.app()
        app.text_area[0].set_value(TRAVELPRO)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        results = app.session_state["analysis"]["classification"]
        self.assertEqual(list(results), list(predict.MODEL_NAMES))
        actual = results["DistilBERT"]
        self.assertEqual(actual["predicted_project_label"], 1)
        self.assertEqual(actual["predicted_label"], "Native Advertising")
        self.assertAlmostEqual(actual["confidence"], 0.6786684393882751, places=5)
        expected = predict.predict_model(TRAVELPRO, "DistilBERT")
        np.testing.assert_allclose(list(actual["probability_by_label"].values()),
                                   list(expected["probability_by_label"].values()), atol=1e-6)
        self.assertEqual(Path(actual["model_path"]), predict.DISTILBERT_DIR)
        probability_tables = [item.value for item in app.dataframe if item.value.columns.tolist() == ["Category", "Probability"]]
        self.assertEqual(len(probability_tables), 3)
        for table in probability_tables:
            self.assertEqual(table["Category"].tolist(), LABELS)
        self.assertEqual(len(app.metric), 11)
        self.assertEqual(app.metric[6].label, "Overall Sentiment")
        app.run()
        self.assertEqual(len(app.metric), 11)
        app.button[1].click().run()
        self.assertEqual(app.text_area[0].value, "")
        self.assertEqual(len(app.metric), 0)

    def test_editing_input_removes_stale_results(self):
        app = self.app()
        app.text_area[0].set_value(TRAVELPRO)
        app.button[0].click().run()
        self.assertGreater(len(app.metric), 0)
        app.text_area[0].set_value("A different passage awaiting analysis.").run()
        self.assertEqual(len(app.metric), 0)

    def test_classifier_failure_is_shown_without_fallback(self):
        app = self.app()
        with patch.object(predict, "predict_text", side_effect=RuntimeError("Unavailable local weights")):
            app.text_area[0].set_value("Text to analyze.")
            app.button[0].click().run()
        self.assertIn("All three final local classifiers are required", app.error[0].value)
        self.assertEqual(len(app.metric), 0)

    def test_sentiment_failure_does_not_change_classification(self):
        app = self.app()
        with patch.object(predict, "predict_sentiment", side_effect=RuntimeError("Sentiment unavailable")):
            app.text_area[0].set_value(TRAVELPRO)
            app.button[0].click().run()
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.metric), 6)
        self.assertEqual(app.session_state["analysis"]["classification"]["DistilBERT"]["predicted_label"], "Native Advertising")
        self.assertIn("Sentiment is unavailable", app.warning[0].value)


if __name__ == "__main__":
    unittest.main()
