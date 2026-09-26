"""Record direct -> prediction module -> Streamlit parity for the exact probe."""
import json
import os
from pathlib import Path
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from streamlit.testing.v1 import AppTest
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from final_project import ROOT, MODEL_PATHS, MAPPING, sha256
from predict import predict_model

TEXT = "This article is brought to you by TravelPro. Book your next vacation with our exclusive summer deals."


def main():
    path = MODEL_PATHS["DistilBERT"]
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
    model.eval()
    encoded = tokenizer(TEXT, return_tensors="pt", max_length=512, truncation=True, return_token_type_ids=False)
    with torch.inference_mode():
        logits = model(**encoded).logits[0]
        probabilities = logits.softmax(-1).numpy()
    internal = int(probabilities.argmax())
    names = MAPPING["internal_id_to_category"]
    direct = {"predicted_class": internal, "predicted_project_label": MAPPING["internal_id_to_project_id"][str(internal)],
              "predicted_label": names[str(internal)], "confidence": float(probabilities[internal]),
              "probability_by_label": {names[str(i)]: float(value) for i, value in enumerate(probabilities)},
              "model_path": str(path), "tokenizer_path": str(path), "input_ids": encoded["input_ids"][0].tolist(),
              "attention_mask": encoded["attention_mask"][0].tolist(), "logits": logits.tolist(),
              "weights_sha256": sha256(path / "model.safetensors"), "tokenizer_sha256": sha256(path / "tokenizer.json")}
    module_result = predict_model(TEXT, "DistilBERT")
    app = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=180).run()
    app.text_area[0].set_value(TEXT)
    app.button[0].click().run()
    if app.exception or app.error:
        raise RuntimeError(f"Streamlit failure: {[e.value for e in app.error]}")
    app_result = app.session_state["analysis"]["classification"]["DistilBERT"]
    for actual in [module_result, app_result]:
        assert actual["model_path"] == direct["model_path"]
        assert actual["weights_sha256"] == direct["weights_sha256"]
        assert actual["tokenizer_path"] == direct["tokenizer_path"]
        assert actual["predicted_class"] == direct["predicted_class"] == 0
        assert actual["number_of_chunks"] == 1
        np.testing.assert_allclose(list(actual["probability_by_label"].values()), probabilities, atol=1e-6, rtol=0)
    output = {"text": TEXT, "direct": direct, "prediction_module": module_result,
              "streamlit": app_result, "parity_passed": True, "probability_absolute_tolerance": 1e-6}
    (ROOT / "results/final_system_audit/travelpro_parity.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
