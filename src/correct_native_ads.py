"""Audit/apply the single annotation-based correction; never generate text or train.

Default is a read-only preview. --apply backs up the original CSV, changes only
category 1 text/group/basis, and saves annotation provenance and the audit.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/final_six_category_dataset.csv"
BACKUP = ROOT / "data/processed/final_six_category_dataset_before_native_ads_correction.csv"
AUDIT_DIR = ROOT / "results/native_ads_correction"
OLD_MANIFEST = ROOT / "results/distilbert_six_category/split_manifest.csv"
BASIS = "Positive Native-Ads label; annotated advertising-containing sentence, source-only boundary repair"


def advertising_sentence(raw):
    """Half-open character offsets are applied BEFORE whitespace normalization."""
    if raw["label"] != 1:
        raise ValueError("Only source-positive advertisements are eligible")
    bounds = []
    for field in ("span", "sen_span"):
        pair = ast.literal_eval(raw[field])
        if not isinstance(pair, tuple) or len(pair) != 2 or any(type(x) is not int for x in pair):
            raise ValueError(f"Invalid {field}: {raw[field]!r}")
        bounds.append(pair)
    (a, b), (c, d) = bounds
    response = raw["response"]
    if not 0 <= c <= a < b <= d <= len(response):
        raise ValueError("Advertising offsets are not nested valid response offsets")
    start = c
    # A few sentence annotations start inside a decimal/URL, or after a street
    # abbreviation. Expand to an earlier natural boundary; never invent text.
    mid_token = c > 0 and response[c-1].isalnum() and response[c].isalnum()
    after_abbreviation = response[c:d].lstrip()[:1].isdigit() and response[:c].rstrip().endswith(".")
    if mid_token or after_abbreviation:
        boundaries = list(re.finditer(r"\n+|[.!?][\"')]*\s+(?=[A-Z])", response[:c]))
        if after_abbreviation and boundaries and boundaries[-1].end() == c:
            boundaries.pop()
        start = boundaries[-1].end() if boundaries else 0
    # Exclude preceding headings/phone/list lines only when they end before
    # the annotated advertisement begins. The complete ad span is retained.
    start = max(start, response.rfind("\n", start, a) + 1)
    assert start <= a and b <= d
    text = " ".join(response[start:d].split())
    if not text or text.casefold() == " ".join(raw["advertisement"].split()).casefold():
        raise ValueError("Empty or brand-only advertising sentence")
    return text, {"span": [a, b], "sen_span": [c, d], "extraction_span": [start, d]}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def row_digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def audit():
    import train_distilbert_six_category as training
    from tokenizers import Tokenizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    if BACKUP.exists():
        raise FileExistsError("Correction has already been applied; refusing a second rebuild")
    fields, old = read_csv(DATASET)
    original_hash = training.sha256_file(DATASET)
    raw_by_location = {}
    counts = Counter()
    raw_files = {}
    for path in sorted((ROOT / "data/raw/Native-Ads").rglob("*.arrow")):
        relative = str(path.relative_to(ROOT))
        raw_files[relative] = training.sha256_file(path)
        with pa.memory_map(str(path), "r") as handle:
            number = 0
            for batch in pa.ipc.open_stream(handle):
                for raw in batch.to_pylist():
                    number += 1
                    counts["total_records"] += 1
                    counts[f"label_{raw['label']}"] += 1
                    if raw["label"] == 1:
                        text, offsets = advertising_sentence(raw)
                        counts["valid_positive_offsets"] += 1
                        counts["equal_span_and_sentence"] += offsets["span"] == offsets["sen_span"]
                        raw_by_location[(relative, str(number))] = (raw, text, offsets)
    new, provenance = [], []
    for index, row in enumerate(old):
        item = dict(row)
        if row["final_label"] == "1":
            raw, text, offsets = raw_by_location[(row["original_file"], row["original_row_number"])]
            if raw["id"] != row["original_id"] or " ".join(raw["response"].split()) != row["text"]:
                raise ValueError("Original Native-Ads provenance does not match the original full response")
            item.update(text=text, text_group_id=hashlib.sha256(text.casefold().encode()).hexdigest(),
                        selection_basis=BASIS)
            provenance.append({"dataset_row": index, "original_id": raw["id"],
                               "original_file": row["original_file"], "original_row_number": row["original_row_number"],
                               "service": raw["service"], "meta_topic": raw["meta_topic"],
                               "query": raw["query"], "advertisement": raw["advertisement"],
                               "label": raw["label"], **offsets, "text": text,
                               "response_sha256": hashlib.sha256(raw["response"].encode()).hexdigest()})
        new.append(item)
    if len(provenance) != 1500:
        raise ValueError("Expected the same 1,500 existing Native-Ads records")
    unchanged_old = [r for r in old if r["final_label"] != "1"]
    unchanged_new = [r for r in new if r["final_label"] != "1"]
    assert unchanged_old == unchanged_new
    for before, after in zip(old, new):
        assert all(before[k] == after[k] for k in fields if k not in {"text", "text_group_id", "selection_basis"})

    tokenizer = Tokenizer.from_file(str(training.base_snapshot() / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()

    def stats(texts):
        lengths = [len(e.ids) for e in tokenizer.encode_batch(texts)]
        return {"count": len(lengths), "mean": float(np.mean(lengths)),
                **dict(zip(["min", "p10", "p25", "median", "p75", "p90", "p95", "max"],
                           np.percentile(lengths, [0, 10, 25, 50, 75, 90, 95, 100]).tolist()))}

    native_indices = [i for i, r in enumerate(new) if r["final_label"] == "1"]
    all_texts = [training.normalized_text(r["text"]) for r in new]
    if len(set(all_texts)) != len(all_texts):
        raise ValueError("Exact normalized duplicate or cross-category text overlap")
    lexical = [training.lexical_text(r["text"]) for r in new]
    lexical_counts = Counter(lexical)
    if any(lexical_counts[lexical[i]] != 1 for i in native_indices):
        raise ValueError("Native-Ads lexical duplicate/cross-category overlap")
    # Search each corrected sentence against the entire final dataset, not just ads.
    features = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(lexical)
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute", radius=0.1, n_jobs=1).fit(features)
    distances, indices = neighbors.radius_neighbors(features[native_indices])
    near_pairs = []
    for i, ds, js in zip(native_indices, distances, indices):
        for distance, j in zip(ds, js):
            if i != j:
                near_pairs.append({"left": i, "right": int(j), "similarity": 1 - float(distance)})
    if near_pairs:
        raise ValueError(f"Near duplicates at character-TFIDF cosine >= 0.90: {near_pairs[:10]}")
    groups, grouping = training.make_groups(new)
    _, manifest = read_csv(OLD_MANIFEST)
    if len(manifest) != len(old):
        raise ValueError("Original split manifest row count mismatch")
    for i, (row, entry) in enumerate(zip(old, manifest)):
        assert int(entry["dataset_row"]) == i and entry["text_group_id"] == row["text_group_id"]
    splits = [entry["split"] for entry in manifest]
    split_counts = training.verify_split(new, groups, splits)
    examples = []
    for topic in sorted({p["meta_topic"] for p in provenance}):
        examples.append(next(p for p in provenance if p["meta_topic"] == topic))
    starts = Counter(" ".join(training.lexical_text(p["text"]).split()[:3]) for p in provenance)
    report = {
        "construction": "Exactly the same 1500 source-positive records and row positions; annotated advertising-containing sentence, with source-only repair of mid-token/abbreviation boundaries and removal of unrelated preceding lines before the ad span. Whitespace normalization only after slicing; the full advertising span is retained. No generated/prefixed text, brand-only text, arbitrary token cropping or variants.",
        "selected_sentence_boundaries_adjusted": sum(p["sen_span"] != p["extraction_span"] for p in provenance),
        "annotation_counts": dict(counts), "original_dataset_sha256": original_hash,
        "categories_2_to_6_unchanged": True, "categories_2_to_6_records": len(unchanged_new),
        "categories_2_to_6_sha256": row_digest(unchanged_new),
        "old_native_token_lengths_including_special_tokens": stats([r["text"] for r in old if r["final_label"] == "1"]),
        "new_native_token_lengths_including_special_tokens": stats([p["text"] for p in provenance]),
        "other_category_token_lengths": {str(i): stats([r["text"] for r in new if r["final_label"] == str(i)]) for i in range(2, 7)},
        "topics": dict(Counter(p["meta_topic"] for p in provenance)),
        "services": dict(Counter(p["service"] for p in provenance)),
        "distinct_advertisements": len({p["advertisement"] for p in provenance}),
        "most_common_three_word_openings": starts.most_common(10),
        "leakage": {"exact_duplicates": 0, "native_lexical_duplicates": 0,
                    "native_near_duplicate_pairs_cosine_at_least_0_90": 0,
                    "cross_category_native_text_overlap": 0, "related_groups_crossing_splits": 0,
                    "all_original_split_assignments_preserved": True},
        "split_counts": split_counts, "grouping": grouping,
        "source_manifest": str(OLD_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": training.sha256_file(OLD_MANIFEST),
        "raw_file_sha256": raw_files, "examples": examples,
        "limitations": ["Cosine screening does not rule out all semantic paraphrases or source/style cues.",
                        "Brands, services and topics are not held out wholesale; queries and related records are grouped.",
                        "The original test and 12 sanity examples informed this correction, so they are regression checks, not a new untouched evaluation.",
                        "No new advertisements were generated; the existing Native-Ads corpus itself contains generated native advertisements."]}
    return fields, old, new, provenance, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    fields, old, new, provenance, report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.apply:
        return
    # A byte-for-byte backup makes this targeted replacement reversible.
    shutil.copy2(DATASET, BACKUP)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = DATASET.with_suffix(".corrected.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(new)
    _, saved = read_csv(temporary)
    assert saved == new
    temporary.replace(DATASET)
    report["corrected_dataset_sha256"] = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    for name, value in [("audit.json", report), ("annotation_provenance.json", provenance)]:
        with (AUDIT_DIR / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print("Applied category-1-only correction; preserved categories 2-6 and original CSV backup.", flush=True)


if __name__ == "__main__":
    main()
