"""Evaluate the 12 fixed user-supplied probes on the completed local model only."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/distilbert_six_category_native_ads_corrected"
RESULTS = ROOT / "results/distilbert_six_category_native_ads_corrected"
PROBES = [
    ("Native Advertising", "Sponsored: Discover our new running shoes and shop the collection today."),
    ("Native Advertising", "This article is brought to you by TravelPro. Book your next vacation with our exclusive summer deals."),
    ("News Satire", "Government announces new department dedicated to explaining why nothing is its responsibility."),
    ("News Satire", "Politicians agree to solve climate change immediately after finishing their next election campaign."),
    ("Propaganda", "Only our movement can save the nation from those who seek to destroy our values."),
    ("Propaganda", "The enemy wants you weak and divided. Stand together and defend our country."),
    ("Manipulation", "After everything I've done for you, you owe me this. If you cared about me, you would agree."),
    ("Manipulation", "Everyone knows you're too sensitive. That never happened; you're just imagining things again."),
    ("News Parody", "Study finds employees are 300% more productive after pretending the internet is down."),
    ("News Parody", "Scientists confirm Monday mornings now officially begin on Sunday night."),
    ("Fabrication", "Researchers at Stanford discovered that humans can survive for six months without sleep."),
    ("Fabrication", "NASA announced that a second moon will become visible from Earth next month."),
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from train_distilbert_native_ads_corrected import validate_correction

    validate_correction()
    if not (RESULTS / "test_metrics.json").exists():
        raise FileNotFoundError("Training and test evaluation must finish first")
    output = RESULTS / "sanity_predictions.json"
    if output.exists():
        raise FileExistsError("Sanity predictions already exist; refusing to overwrite")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL), local_files_only=True)
    model.eval()
    mapping = read_json(MODEL / "label_mapping.json")
    rows = []
    with torch.inference_mode():
        for number, (intended, text) in enumerate(PROBES, 1):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            probabilities = model(**inputs).logits.softmax(dim=-1)[0]
            index = int(probabilities.argmax())
            category = mapping["internal_id_to_category"][str(index)]
            assert category == model.config.id2label[index]
            row = {"number": number, "text": text, "intended_category": intended,
                   "predicted_project_label": mapping["internal_id_to_project_id"][str(index)],
                   "predicted_category": category, "confidence": float(probabilities[index]),
                   "matches": intended == category}
            rows.append(row)
            print(json.dumps(row), flush=True)
    with (MODEL / "model.safetensors").open("rb") as handle:
        weights_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    result = {"model": str(MODEL), "model_sha256": weights_hash,
              "correct": sum(r["matches"] for r in rows), "total": len(rows),
              "errors": [r["number"] for r in rows if not r["matches"]], "predictions": rows}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    audit = read_json(ROOT / "results/native_ads_correction/audit.json")
    validation = read_json(RESULTS / "validation_metrics.json")
    test = read_json(RESULTS / "test_metrics.json")
    with (RESULTS / "test_per_class_metrics.csv").open(encoding="utf-8", newline="") as handle:
        classes = list(csv.DictReader(handle))
    with (RESULTS / "test_confusion_matrix.csv").open(encoding="utf-8", newline="") as handle:
        matrix = list(csv.reader(handle))
    lines = ["# Native Advertising correction — completed run", "",
             "Only category 1 construction changed. All 6,959 category 2–6 rows, labels, metadata and every original split assignment are unchanged. Original model/results and a byte-for-byte dataset backup are retained. No API, generated training text, hyperparameter search, or Streamlit changes.", "",
             "## Annotation interpretation and construction", "",
             "`span` and `sen_span` are string-encoded `(start, end)` half-open character offsets into `response`: advertising passage and containing sentence, respectively. All 6,041 positive annotations are valid and nested; 4,626 have equal spans. The source annotations have occasional boundary errors.", "",
             audit["construction"], "",
             "32 of the 1,500 excerpts have source-only sentence-boundary adjustments. Original and extraction offsets, source ID/file/row, query, advertisement, service/topic and response hash are saved in `../native_ads_correction/annotation_provenance.json`.", "",
             "The same 1,500 records cover 10 topics, 583 advertisements, 904 Bing and 596 YouChat responses. The most common three-word opening occurs in 63/1,500 records (4.2%); no prefix/template was added.", "",
             "## Token lengths (DistilBERT, including special tokens)", "",
             "| Statistic | Old | New |", "|---|---:|---:|"]
    for name in ["min", "mean", "p10", "p25", "median", "p75", "p90", "p95", "max"]:
        old = audit["old_native_token_lengths_including_special_tokens"][name]
        new = audit["new_native_token_lengths_including_special_tokens"][name]
        lines.append(f"| {name} | {old:.2f} | {new:.2f} |")
    lines += ["", "## Ten actual corrected examples", ""]
    for i, sample in enumerate(audit["examples"], 1):
        lines.append(f"{i}. **{sample['meta_topic']} — {sample['original_id']}**: {sample['text']}")
    lines += ["", "## Overlap and limitations", "",
              "Zero exact duplicates, Native-Ads lexical duplicates, Native-Ads cross-category text matches, or Native-Ads near-duplicate pairs at character TF-IDF cosine ≥ 0.90. Zero related groups crossing partitions. Original query/provenance/template groups and per-row partitions are preserved. Category 1 retains 1,050 train / 225 validation / 225 test records; totals remain 5,921 / 1,269 / 1,269.", ""]
    lines += [f"- {item}" for item in audit["limitations"]]
    lines += ["", "## Training", "",
              "One run, same original pretrained DistilBERT revision, seed 42, 3 epochs, max length 512, learning rate 2e-5, weight decay 0.01, train/eval batch sizes 8/16, accumulation 2, warmup 10%, linear schedule, AdamW, original train-only class weights. Best checkpoint selected by validation macro F1. See training_plan.json and training_history.json.", "",
              "## Evaluation", "", "| Metric | Validation | Test |", "|---|---:|---:|"]
    for name in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]:
        lines.append(f"| {name} | {validation[name]:.9f} | {test[name]:.9f} |")
    lines += ["", "| Category | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for row in classes:
        lines.append(f"| {row['category']} | {float(row['precision']):.6f} | {float(row['recall']):.6f} | {float(row['f1']):.6f} | {int(float(row['support']))} |")
    lines += ["", "Confusion matrix: rows = actual, columns = predicted.", "",
              "| " + " | ".join(matrix[0]) + " |", "|" + "---|" * len(matrix[0])]
    lines += ["| " + " | ".join(row) + " |" for row in matrix[1:]]
    mistakes = sum(int(value) for i, row in enumerate(matrix[1:]) for j, value in enumerate(row[1:]) if i != j)
    lines += ["", f"Test mistakes: {mistakes} / {test['records']}.", "",
              "## Fixed 12-example sanity check", "",
              f"Correct: **{result['correct']} / 12**. Confidence is softmax probability, not calibrated correctness.", "",
              "| # | Intended | Project label | Predicted | Confidence | Match |", "|---|---|---:|---|---:|---|"]
    for row in rows:
        lines.append(f"| {row['number']} | {row['intended_category']} | {row['predicted_project_label']} | {row['predicted_category']} | {row['confidence']:.2%} | {'Yes' if row['matches'] else 'No'} |")
    lines += ["", "Exact probe texts:", ""]
    lines += [f"{r['number']}. {r['text']}" for r in rows]
    lines += ["", f"Errors: {result['errors']}.", "", "Native Advertising probes:", ""]
    lines += [f"- #{r['number']}: **{r['predicted_category']}**, project label {r['predicted_project_label']}, confidence **{r['confidence']:.2%}**, {'correct' if r['matches'] else 'incorrect'}." for r in rows[:2]]
    with (RESULTS / "correction_report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"correct": result["correct"], "total": 12, "errors": result["errors"]}), flush=True)


if __name__ == "__main__":
    main()
