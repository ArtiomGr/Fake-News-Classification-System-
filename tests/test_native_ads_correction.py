"""Annotation extraction checks; no model loading or training."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import correct_native_ads as correction


class NativeAdsExtractionTests(unittest.TestCase):
    def raw(self, response, ad_start, sentence_start):
        return {"label": 1, "advertisement": "Brand", "response": response,
                "span": str((ad_start, len(response))),
                "sen_span": str((sentence_start, len(response)))}

    def test_mid_decimal_recovers_real_sentence(self):
        response = "First sentence. Buy this game for EUR54.99 and enjoy a remarkable adventure."
        raw = self.raw(response, response.index("and enjoy"), response.index("99") + 1)
        text, offsets = correction.advertising_sentence(raw)
        self.assertEqual(text, response[len("First sentence. "):])
        self.assertEqual(offsets["extraction_span"][0], len("First sentence. "))

    def test_unrelated_heading_before_ad_is_excluded(self):
        response = "Healthcare.gov Plans\nConsider Brand for comprehensive coverage."
        raw = self.raw(response, response.index("Brand"), response.index("ov Plans"))
        text, _ = correction.advertising_sentence(raw)
        self.assertEqual(text, "Consider Brand for comprehensive coverage.")

    def test_advertisement_is_never_used_as_text(self):
        response = "Explore Brand for comfortable footwear and convenient delivery."
        text, _ = correction.advertising_sentence(self.raw(response, 0, 0))
        self.assertEqual(text, response)

    def test_bad_or_negative_annotations_rejected(self):
        raw = self.raw("Choose Brand today.", 0, 0)
        raw["label"] = 0
        with self.assertRaises(ValueError):
            correction.advertising_sentence(raw)
        raw["label"] = 1
        raw["span"] = "(-1, 10)"
        with self.assertRaises(ValueError):
            correction.advertising_sentence(raw)

    @unittest.skipUnless((correction.AUDIT_DIR / "audit.json").exists(), "Correction not applied")
    def test_applied_dataset_and_source_traceability(self):
        import hashlib
        import pyarrow as pa
        from train_distilbert_native_ads_corrected import validate_correction
        provenance = json.loads((correction.AUDIT_DIR / "annotation_provenance.json").read_text(encoding="utf-8"))
        required = [correction.BACKUP, *[correction.ROOT / p["original_file"] for p in provenance]]
        if any(not path.is_file() for path in required):
            self.skipTest("Optional raw-source/backup audit requires local files intentionally excluded from Git")
        validate_correction()
        _, rows = correction.read_csv(correction.DATASET)
        source_rows = {}
        for name in {p["original_file"] for p in provenance}:
            with pa.memory_map(str(correction.ROOT / name), "r") as handle:
                source_rows[name] = [row for batch in pa.ipc.open_stream(handle) for row in batch.to_pylist()]
        for record in provenance:
            raw = source_rows[record["original_file"]][int(record["original_row_number"]) - 1]
            self.assertEqual(raw["id"], record["original_id"])
            self.assertEqual(hashlib.sha256(raw["response"].encode()).hexdigest(), record["response_sha256"])
            text, offsets = correction.advertising_sentence(raw)
            self.assertEqual(text, rows[record["dataset_row"]]["text"])
            self.assertEqual(offsets["extraction_span"], record["extraction_span"])


if __name__ == "__main__":
    unittest.main()
