"""Run the 12 fixed qualitative probes through the completed final system.

These probes are not training data or an independent performance benchmark.
"""
import csv
import json
from final_project import ROOT
from evaluate_native_ads_correction import PROBES
from predict import predict_text, predict_sentiment


def main():
    output = ROOT / "results/final_model_comparison"
    output.mkdir(parents=True, exist_ok=True)
    detailed, tabular = [], []
    for number, (intended, text) in enumerate(PROBES, 1):
        results = predict_text(text)
        sentiment = predict_sentiment(text)
        detailed.append({"number": number, "text": text, "intended_category": intended,
                         "classification": results, "sentiment": sentiment})
        for name, result in results.items():
            row = {"number": number, "model": name, "intended_category": intended,
                   "predicted_project_label": result["predicted_project_label"],
                   "predicted_category": result["predicted_label"], "confidence": result["confidence"],
                   "matches_intended_category": result["predicted_label"] == intended}
            tabular.append(row)
            print(json.dumps(row), flush=True)
    (output / "representative_predictions.json").write_text(json.dumps(detailed, indent=2) + "\n", encoding="utf-8")
    with (output / "representative_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tabular[0]))
        writer.writeheader()
        writer.writerows(tabular)
    for name in results:
        correct = sum(row["matches_intended_category"] for row in tabular if row["model"] == name)
        print(f"{name}: {correct}/12 matches to intended categories", flush=True)


if __name__ == "__main__":
    main()
