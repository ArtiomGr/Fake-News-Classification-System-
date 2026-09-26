"""Prepare unlabeled data, deduplicating within each source only.

Run from any directory: python src/prepare_combined_dataset.py
Only data/processed/combined_unlabeled.csv is written; an existing output is
never overwritten. Labels are copied from the inputs, not inferred or mapped.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/combined_unlabeled.csv"
RANDOM_STATE = 42
FAKEDDIT_LIMIT = 30_000
FIELDS = [
    "text", "source", "original_label", "original_subtype",
    "label_2way", "label_3way", "label_6way", "source_split", "final_label",
]
SOURCES = ["manipulation", "native_ads", "propaganda", "fakeddit"]


def normalize_text(value: str | None) -> str:
    """Collapse Unicode whitespace; preserve case, punctuation, and wording."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"Expected text or null, got {type(value).__name__}")
    return " ".join(value.split())


def preserve_label(value: object) -> str:
    """Represent original labels in CSV, including lowercase JSON booleans."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return json.dumps(value)
    return str(value)


def metadata_json(values: dict) -> str:
    """Keep named subtype information in one reversible JSON-valued column."""
    available = {key: value for key, value in values.items()
                 if value is not None and value != ""}
    return json.dumps(available, ensure_ascii=False, sort_keys=True) if available else ""


def read_delimited(path: Path, delimiter: str, required: set[str]):
    """Read literal CSV/TSV strings without treating words such as NA as null."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: malformed delimited record near line {line}")
            yield row


def input_records():
    """Yield (file, source, text, label, subtype, metadata) in fixed order.

    Arrow shards are discovered by their real filenames, ignoring potentially
    stale state.json entries. Fakeddit's two input files form one sampling pool.
    """
    path = ROOT / "data/raw/Manipulation/manipulational_conversation.jsonl"
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            if messages is None:
                messages = []
            if not isinstance(messages, list):
                raise TypeError(f"{path}: messages must be a list or null")
            parts = [message["text"] for message in messages]
            if any(part is not None and not isinstance(part, str) for part in parts):
                raise TypeError(f"{path}: message text must be a string or null")
            text = "\n".join(part for part in parts if part is not None)
            yield (path, "manipulation", text, row["is_manipulation"],
                   row["manipulation_type"], {})

    for split in ["train", "validation", "test"]:
        folder = ROOT / "data/raw/Native-Ads" / split
        paths = sorted(folder.rglob("*.arrow"))
        if not paths:
            raise FileNotFoundError(f"No Arrow files found in {folder}")
        for path in paths:
            with pa.memory_map(str(path), "r") as handle:
                reader = pa.ipc.open_stream(handle)
                required = {"response", "label"}
                if not required.issubset(reader.schema.names):
                    raise ValueError(f"{path}: missing response or label")
                for batch in reader:
                    for row in batch.to_pylist():
                        subtype = metadata_json({
                            "advertisement": row.get("advertisement"),
                            "meta_topic": row.get("meta_topic"),
                        })
                        yield (path, "native_ads", row["response"], row["label"],
                               subtype, {"source_split": split})

    path = ROOT / "data/raw/propaganda_dataset/propaganda_dataset.csv"
    for row in read_delimited(path, ",", {"text", "label"}):
        yield path, "propaganda", row["text"], row["label"], "", {}

    required = {"title", "6_way_label", "2_way_label", "3_way_label"}
    for filename, split in [("all_train.tsv", "train"), ("all_test_public.tsv", "test")]:
        path = ROOT / "data" / filename
        for row in read_delimited(path, "\t", required):
            metadata = {
                "label_2way": row["2_way_label"],
                "label_3way": row["3_way_label"],
                "label_6way": row["6_way_label"],
                "source_split": split,
            }
            yield path, "fakeddit", row["title"], row["6_way_label"], "", metadata


def sample_fakeddit(groups: dict[str, list[dict]]) -> list[dict]:
    """Sample equally across observed labels using random_state=42.

    Allocate one slot per class per round. If a class runs out, redistribute
    its remaining slots across the others. Sorted labels resolve remainders.
    With six sufficiently populated classes, each receives 5,000 slots.
    Sampling happens after normalization and within-source deduplication.
    """
    labels = sorted(groups)
    quotas = dict.fromkeys(labels, 0)
    remaining = min(FAKEDDIT_LIMIT, sum(len(rows) for rows in groups.values()))
    while remaining:
        for label in labels:
            if quotas[label] < len(groups[label]):
                quotas[label] += 1
                remaining -= 1
                if not remaining:
                    break
    rng = random.Random(RANDOM_STATE)
    selected = []
    for label in labels:
        # Keep selected rows in input order within each class.
        indices = sorted(rng.sample(range(len(groups[label])), quotas[label]))
        selected.extend(groups[label][index] for index in indices)
    return selected


def cross_source_summary(text_sources: dict[str, set[str]]) -> tuple[int, int]:
    """Count shared distinct texts and records involved, without removing any.

    Within-source deduplication leaves one record per text per source. A text
    present in three sources counts as one shared text and three records.
    """
    shared = [len(sources) for sources in text_sources.values() if len(sources) > 1]
    return len(shared), sum(shared)


def main() -> None:
    """Load, normalize, deduplicate, sample, write once, and report counts."""
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {OUTPUT}")

    counts = {source: Counter() for source in SOURCES}
    file_counts = Counter()
    seen = {source: {} for source in SOURCES}
    text_sources = defaultdict(set)
    retained = []
    fakeddit_groups = defaultdict(list)

    for path, source, raw_text, label, subtype, metadata in input_records():
        counts[source]["loaded"] += 1
        file_counts[str(path.relative_to(ROOT))] += 1
        text = normalize_text(raw_text)
        if not text:
            counts[source]["missing"] += 1
            continue
        original_label = preserve_label(label)
        original_subtype = "" if subtype is None else subtype
        record = dict.fromkeys(FIELDS, "")
        record.update(text=text, source=source, original_label=original_label,
                      original_subtype=original_subtype)
        record.update(metadata)
        identity = tuple(record[field] for field in FIELDS if field not in {"text", "source"})
        # Splits belong to the same source: keep the first occurrence across
        # that source's splits, preserving its labels and split provenance.
        if text in seen[source]:
            counts[source]["duplicates"] += 1
            if seen[source][text] != identity:
                counts[source]["duplicate_metadata_differences"] += 1
            continue
        seen[source][text] = identity
        # Observation only: shared text in another source is never filtered.
        text_sources[text].add(source)
        if source == "fakeddit":
            if not original_label:
                raise ValueError("A nonempty Fakeddit title has no 6_way_label")
            fakeddit_groups[original_label].append(record)
        else:
            retained.append(record)

    cross_before = cross_source_summary(text_sources)
    sampled = sample_fakeddit(fakeddit_groups)
    counts["fakeddit"]["sampling_excluded"] = (
        sum(len(rows) for rows in fakeddit_groups.values()) - len(sampled)
    )
    retained.extend(sampled)
    distributions = {source: Counter() for source in SOURCES}
    retained_text_sources = defaultdict(set)
    for record in retained:
        source = record["source"]
        counts[source]["retained"] += 1
        distributions[source][record["original_label"]] += 1
        retained_text_sources[record["text"]].add(source)
    cross_after = cross_source_summary(retained_text_sources)
    for source, stats in counts.items():
        assert stats["loaded"] == sum(stats[key] for key in
            ["missing", "duplicates", "sampling_excluded", "retained"]), source

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against an output appearing mid-run.
    with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(retained)

    print("Records loaded by input file:")
    for filename, count in file_counts.items():
        print(f"  {filename}: {count:,}")
    print("\nSource report (within-source duplicates only; first occurrence wins):")
    print(f"{'Source':<15} {'Loaded':>10} {'Missing':>10} {'Duplicates':>12} "
          f"{'Not sampled':>12} {'Retained':>10}")
    for source, stats in counts.items():
        print(f"{source:<15} {stats['loaded']:>10,} {stats['missing']:>10,} "
              f"{stats['duplicates']:>12,} {stats['sampling_excluded']:>12,} "
              f"{stats['retained']:>10,}")
    print("\nOriginal label distributions in the retained output:")
    for source, distribution in distributions.items():
        print(f"  {source}: {dict(sorted(distribution.items()))}")
    print("\nRemoved within-source duplicates with differing label/subtype/split metadata:")
    for source, stats in counts.items():
        print(f"  {source}: {stats['duplicate_metadata_differences']:,}")
    print("\nCross-source duplicates (reported only; not removed):")
    for stage, (texts, records) in [
        ("After within-source deduplication, before sampling", cross_before),
        ("In final output, after sampling", cross_after),
    ]:
        print(f"  {stage}: {texts:,} distinct shared texts; {records:,} records involved")
    print(f"\nFakeddit sampling random_state: {RANDOM_STATE}")
    print(f"Total final records: {len(retained):,}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
