"""Build three approved label pools and three UNLABELED review pools offline.

This is an intermediate dataset, not a balanced six-category training set.
No model, API, synthetic generation, or automatic Fakeddit labeling is used.
Run only after approval: python src/build_reliable_dataset.py
All inputs are read-only; existing output files are never overwritten.

Candidate policy is deliberately conservative: require hasImage=False and an
empty image_url, and reject inconsistent recovered metadata. Absence of an
image flag does not establish text independence or category membership.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
INPUT = PROCESSED / "combined_unlabeled.csv"
OUTPUTS = {
    "reliable": PROCESSED / "reliable_dataset.csv",
    "satire": PROCESSED / "news_satire_candidates.csv",
    "propaganda": PROCESSED / "propaganda_fakeddit_candidates.csv",
    "fabrication": PROCESSED / "fabrication_candidates.csv",
}
MAPPINGS = {
    ("native_ads", "1"): (1, "Native Advertising"),
    ("propaganda", "1"): (3, "Propaganda"),
    ("manipulation", "true"): (4, "Manipulation"),
}
EXTRA_FIELDS = [
    "final_category", "subreddit", "title", "2_way_label", "3_way_label",
    "6_way_label", "hasImage", "image_url", "raw_ids", "raw_match_count",
    "raw_source_splits", "text_group_id", "original_holdout",
    "needs_manual_review", "selection_reason",
]
SATIRE_SUBREDDITS = {"theonion", "satire", "waterfordwhispersnews"}


def normalize(text: str) -> str:
    """Use exactly the whitespace normalization used by preprocessing."""
    return " ".join(text.split())


def read_csv(path: Path, delimiter: str, required: set[str]):
    """Yield string values literally, without NA inference or label coercion."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"Invalid or missing columns: {path}")
        for index, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed row near {path}:{index}")
            yield row


def combined_key(row: dict) -> tuple:
    """Match text AND original split AND all three labels, never position alone."""
    return (normalize(row["text"]), row["source_split"], row["label_2way"],
            row["label_3way"], row["label_6way"])


def recover_metadata(rows: list[dict]) -> tuple[dict, dict]:
    """Recover all matching raw rows and detect conflicting metadata.

    Different IDs or original whitespace do not themselves imply a conflict.
    Different subreddit, image flag, or image-URL presence do. Preserve the
    first raw title/URL and all IDs for audit; never silently choose between
    conflicting candidate-selection metadata. Also detect raw train/test text
    overlap independently of labels to flag original held-out text.
    """
    wanted = {combined_key(row) for row in rows if row["source"] == "fakeddit"}
    wanted_texts = {key[0].casefold() for key in wanted}
    matches = {}
    raw_splits = defaultdict(set)
    required = {"title", "subreddit", "2_way_label", "3_way_label", "6_way_label",
                "hasImage", "image_url", "id"}
    for filename, split in [("all_train.tsv", "train"), ("all_test_public.tsv", "test")]:
        for raw in read_csv(ROOT / "data" / filename, "\t", required):
            text = normalize(raw["title"])
            if text.casefold() in wanted_texts:
                raw_splits[text.casefold()].add(split)
            key = (text, split, raw["2_way_label"], raw["3_way_label"], raw["6_way_label"])
            if key not in wanted:
                continue
            entry = matches.setdefault(key, {
                "first": raw, "signatures": set(), "ids": set(), "count": 0,
            })
            entry["count"] += 1
            entry["ids"].add(raw["id"])
            entry["signatures"].add((raw["subreddit"].strip().lower(),
                                      raw["hasImage"].strip().lower(),
                                      bool(raw["image_url"].strip())))
    missing = wanted - matches.keys()
    if missing:
        raise ValueError(f"Cannot recover {len(missing)} Fakeddit records; refusing partial output")
    return matches, raw_splits


def candidate_pool(row: dict, entry: dict) -> tuple[str, str] | None:
    """Choose review pools only; these rules never assign final labels."""
    if len(entry["signatures"]) != 1:
        return None
    subreddit, has_image, image_url_present = next(iter(entry["signatures"]))
    if has_image != "false" or image_url_present:
        return None
    label = row["label_6way"]
    if label == "1" and subreddit in SATIRE_SUBREDDITS:
        return "satire", "Satire provenance; review text and distinguish news satire from news parody"
    if label == "5" and row["label_3way"] == "1" and subreddit == "propagandaposters":
        return "propaganda", "Propaganda provenance; verify text stands alone, not just a poster caption"
    if label == "5" and subreddit == "fakefacts":
        return "fabrication", "Fakefacts provenance; verify factual form and distinguish fabrication from jokes"
    # Exclude album covers, visual alteration/false-connection pools, simulator
    # posts, and corrected clickbait: none establishes a textual fabrication.
    return None


def main() -> None:
    """Build auditable pools without balancing, splitting, or training."""
    existing = [str(path) for path in OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing}")
    required = {"text", "source", "original_label", "original_subtype", "label_2way",
                "label_3way", "label_6way", "source_split", "final_label"}
    rows = list(read_csv(INPUT, ",", required))
    if not rows:
        raise ValueError("Combined input is empty")
    fields = list(rows[0])
    if set(fields) & set(EXTRA_FIELDS):
        raise ValueError("Input already contains output metadata columns")
    if any(not normalize(row["text"]) or row["final_label"] != "" for row in rows):
        raise ValueError("Expected nonempty text and empty final_label throughout input")
    if any(row["source"] == "fakeddit" and row["original_label"] != row["label_6way"]
           for row in rows):
        raise ValueError("Fakeddit original_label and label_6way disagree")
    matches, raw_splits = recover_metadata(rows)
    outputs = {name: [] for name in OUTPUTS}
    possible = []
    audit = Counter()
    for index, row in enumerate(rows):
        mapping = MAPPINGS.get((row["source"], row["original_label"]))
        record = {**row, **dict.fromkeys(EXTRA_FIELDS, "")}
        # Group ID ignores case as well as whitespace, so future partitions can
        # keep those variants together even though exact dedup preserves case.
        group_text = normalize(row["text"]).casefold()
        record["text_group_id"] = hashlib.sha256(group_text.encode("utf-8")).hexdigest()
        record["original_holdout"] = row["source_split"] in {"validation", "test"}
        if mapping:
            pool = "reliable"
            record.update(final_label=mapping[0], final_category=mapping[1],
                          needs_manual_review=False,
                          selection_reason="User-approved positive source-label mapping")
        elif row["source"] == "fakeddit":
            entry = matches[combined_key(row)]
            if len(entry["signatures"]) != 1:
                audit["ambiguous_metadata_excluded"] += 1
            selection = candidate_pool(row, entry)
            if selection is None:
                continue
            pool, reason = selection
            raw = entry["first"]
            for name in ["subreddit", "title", "2_way_label", "3_way_label",
                         "6_way_label", "hasImage", "image_url"]:
                record[name] = raw[name]
            splits = sorted(raw_splits[group_text])
            record.update(raw_ids=json.dumps(sorted(entry["ids"])),
                          raw_match_count=entry["count"],
                          raw_source_splits=json.dumps(splits),
                          original_holdout="test" in splits,
                          needs_manual_review=True, selection_reason=reason)
            # Both final_label and final_category remain empty for candidates.
        else:
            continue
        possible.append((index, pool, record))

    # Conflicting destinations/labels for identical normalized text are held
    # out entirely, rather than silently choosing a training label or pool.
    destinations = defaultdict(set)
    heldout_groups = set()
    for _, pool, record in possible:
        destinations[normalize(record["text"])].add((pool, record["final_label"]))
        if record["original_holdout"]:
            heldout_groups.add(record["text_group_id"])
    seen = set()
    for _, pool, record in possible:
        text = normalize(record["text"])
        if len(destinations[text]) > 1:
            audit["conflicting_text_records_excluded"] += 1
            continue
        if text in seen:
            audit["exact_duplicates_removed"] += 1
            continue
        seen.add(text)
        record["original_holdout"] = record["text_group_id"] in heldout_groups
        outputs[pool].append(record)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for pool, path in OUTPUTS.items():
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields + EXTRA_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(outputs[pool])
        print(f"{path}: {len(outputs[pool]):,} records")
    print("Reliable labels:", dict(Counter(row["final_category"] for row in outputs["reliable"])))
    print("Exclusion audit:", dict(audit))
    print("Candidate final labels/categories are empty; all require manual review.")
    print("No balancing or train/validation/test split was performed.")
    print("Before training: respect original holdouts; group by text_group_id and")
    print("audit near duplicates, shared templates, and related source records.")


if __name__ == "__main__":
    main()
