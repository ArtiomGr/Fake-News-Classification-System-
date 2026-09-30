"""Focused USER/ADMIN interactions and the actual selected prediction path."""
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import final_decision
import predict
import database
import sentiment

TRAVELPRO = 'This article is brought to you by TravelPro. Book your next vacation with our exclusive summer deals.'


class FinalDashboardTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.history_path = Path(directory.name) / 'history.json'
        original_save = database.save_analysis
        saver = patch.object(database, 'save_analysis',
                             side_effect=lambda text, result: original_save(text, result, self.history_path))
        saver.start()
        self.addCleanup(saver.stop)
        for module, name in ((final_decision, 'predict_final'),
                             (final_decision, 'vader_features'),
                             (sentiment, 'analyze_vader_sentiment'),
                             (predict, 'predict_sentiment')):
            guard = patch.object(module, name, side_effect=AssertionError('USER must use only the classifier'))
            mocked = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(mocked.assert_not_called)

    def app(self):
        app = AppTest.from_file(str(ROOT / 'app/app.py'), default_timeout=180).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def analyze(self, app, text=TRAVELPRO):
        app.text_area(key='input_text').set_value(text)
        app.button(key='analyze').click().run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_user_starts_clean_without_model_loading(self):
        with patch.object(predict, '_load_classifier', side_effect=AssertionError('No eager model loading')):
            app = self.app()
        self.assertEqual(app.radio(key='view').value, 'USER')
        self.assertEqual([b.label for b in app.button], ['Analyze', 'Clear'])
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.dataframe), 0)

    def test_empty_input_does_not_invoke_inference(self):
        app = self.app()
        with patch.object(predict, 'predict_model') as call:
            self.analyze(app, '  ')
            call.assert_not_called()
        self.assertEqual(app.warning[0].value, 'Please enter text before running the analysis.')
        self.assertEqual(len(app.metric), 0)

    def test_actual_final_path_one_category_confidence_and_clear(self):
        app = self.app()
        self.analyze(app)
        expected = predict.predict_model(TRAVELPRO, 'DistilBERT')
        self.assertEqual(len(app.error), 0)
        result = app.session_state['analysis']['result']
        self.assertEqual(result['system'], 'classifier_alone')
        self.assertEqual(result['classifier'], 'DistilBERT')
        self.assertEqual(result['predicted_label'], 'Native Advertising')
        self.assertAlmostEqual(result['confidence'], expected['confidence'], places=10)
        self.assertEqual(result['probability_by_label'], expected['probability_by_label'])
        self.assertEqual(result['confidence'], expected['confidence'])
        self.assertNotIn('sentiment', app.session_state['analysis'])
        rows = database.get_recent_analyses(database_path=self.history_path)
        self.assertEqual(rows[0]['confidence'], expected['confidence'])
        self.assertEqual(rows[0]['system'], 'classifier_alone')
        self.assertEqual(set(rows[0]), {'id', 'analyzed_at', 'text_sha256', 'text_length',
                                      'system', 'classifier', 'predicted_label', 'confidence',
                                      'number_of_chunks', 'inference_seconds'})
        self.assertEqual(len(app.metric), 2)
        self.assertEqual(app.metric[0].label, 'Final category')
        self.assertEqual(app.metric[1].label, 'Model confidence')
        self.assertEqual(app.metric[1].value, f"{expected['confidence']:.2%}")
        self.assertEqual(len(app.dataframe), 0)
        app.run()
        self.assertEqual(len(app.metric), 2)
        app.button(key='clear').click().run()
        self.assertEqual(app.text_area(key='input_text').value, '')
        self.assertEqual(len(app.metric), 0)

    def test_edit_removes_stale_result(self):
        app = self.analyze(self.app())
        app.text_area(key='input_text').set_value('A different article.').run()
        self.assertEqual(len(app.metric), 0)

    def test_artifact_version_change_invalidates_displayed_result(self):
        app = self.analyze(self.app())
        signature = tuple(predict.artifact_fingerprint()) + (('registry', 'changed', 1, 1),)
        with patch.object(predict, 'artifact_fingerprint', return_value=signature):
            app.run()
        self.assertEqual(len(app.metric), 0)

    def test_prediction_failure_clears_old_result_without_fallback(self):
        app = self.analyze(self.app())
        with patch.object(predict, 'predict_model', side_effect=RuntimeError('Unavailable classifier')) as call:
            self.analyze(app)
            call.assert_called_once_with(TRAVELPRO, 'DistilBERT')
        self.assertIn('Analysis is currently unavailable', app.error[0].value)
        self.assertEqual(len(app.metric), 0)

    def test_unusable_tokens_show_input_error(self):
        app = self.app()
        self.analyze(app, '\u200b\u200c')
        self.assertEqual(len(app.metric), 0)
        self.assertIn('Try a passage with readable words', app.error[0].value)

    def test_admin_saved_evidence_without_inference(self):
        app = self.app()
        with patch.object(predict, 'predict_model', side_effect=AssertionError('ADMIN must not infer')):
            app.radio(key='view').set_value('ADMIN').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.tabs), 6)
        self.assertEqual(len(app.text_area), 0)
        comparison = next(t.value for t in app.dataframe if t.value.columns[0] == 'Classifier' and len(t.value) == 9)
        self.assertEqual(set(comparison['Partition']), {'TRAIN', 'Validation', 'Test'})
        self.assertEqual(set(comparison['Classifier']), {'BERT', 'DistilBERT', 'Logistic Regression'})
        baseline = json.loads((ROOT / 'results/final_decision_v1/test_results.json').read_text())
        metric = next(m for m in app.metric if m.label == 'Phase 1–5 tests passed')
        self.assertEqual(metric.value, str(baseline['tests_run']))
        text = '\n'.join(m.value for m in [*app.markdown, *app.caption, *app.warning, *app.success])
        for phrase in ('No statistically supported incremental VADER benefit', 'not an untouched independent',
                       'historical Phase 1–5', 'not authenticated access roles', 'probability_only'):
            self.assertIn(phrase, text)

    def test_confusion_controls_and_return_to_user(self):
        app = self.app()
        app.radio(key='view').set_value('ADMIN').run()
        app.radio(key='matrix_family').set_value('Decision systems').run()
        app.selectbox(key='matrix_model').set_value('B · Probability-only Decision Layer').run()
        app.selectbox(key='matrix_split').set_value('Validation · nested OOF').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        matrix = next(t.value for t in app.dataframe if t.value.shape == (6, 6))
        self.assertEqual(int(matrix.to_numpy().sum()), 1269)
        app.radio(key='view').set_value('USER').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.text_area), 1)


if __name__ == '__main__':
    unittest.main()
