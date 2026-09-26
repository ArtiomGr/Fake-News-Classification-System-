"""Build six offline heuristic label pools; never invent records to meet quotas.

Inputs remain read-only. Run with --preview for counts without output files,
or without arguments to write the final CSV and report. Existing outputs are
never overwritten. No APIs, models, network, or synthetic generation are used.
The output is source-supervised/heuristic data, not human-verified ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa

from correct_native_ads import advertising_sentence, BASIS as NATIVE_ADS_BASIS


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"
COMBINED = PROCESSED / "combined_unlabeled.csv"
OUTPUT = PROCESSED / "final_six_category_dataset.csv"
REPORT = PROCESSED / "final_dataset_report.txt"
TARGET = 1500
RANDOM_STATE = 42
CATEGORIES = {1: "Native Advertising", 2: "News Satire", 3: "Propaganda",
              4: "Manipulation", 5: "News Parody", 6: "Fabrication"}
FIELDS = [
    "text", "final_label", "final_category", "source", "original_label",
    "original_subtype", "source_split", "subreddit", "domain", "original_id",
    "label_2way", "label_3way", "label_6way", "original_title", "hasImage",
    "image_url", "author", "created_utc", "linked_submission_id",
    "service", "query", "original_file", "original_row_number",
    "text_group_id", "observed_source_splits", "matching_original_ids",
    "selection_basis", "heuristic_priority",
]
SATIRE_SOURCES = {"theonion", "satire", "waterfordwhispersnews"}
WANTED_SOURCES = SATIRE_SOURCES | {"propagandaposters", "fakefacts", "fakehistoryporn"}


def pattern(expression: str) -> re.Pattern:
    return re.compile(expression, re.IGNORECASE)


WORDS = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
META = pattern(r"^(?:\[?(?:meta|mod|discussion|announcement|request|weekly|monthly)\b|"
               r"please\s|can someone\s|does anyone\s|anyone know\s|help me\s)")
URL = pattern(r"https?://|www\.")
# These first five screens reproduce the earlier 2,636-record parody pool.
VISUAL = pattern(r"\b(?:photo(?:graph)?s?|pictured|image|picture|colori[sz]ed|colourised|"
                 r"painting|portrait|meme|screenshot|thumbnail|photoshop(?:ped)?)\b")
NEWS = pattern(r"\b(?:report|study|scientists?|researchers?|doctors?|experts?|officials?|"
               r"announces?|announced|reveals?|finds?|confirms?|warns?|advises?|urges?|"
               r"according to|breaking|survey|poll)\b")
PUBLIC = pattern(r"\b(?:trump|clinton|obama|biden|bush|sanders|republicans?|democrats?|"
                 r"congress|senate|senators?|president|government|politic\w*|election|voters?|"
                 r"brexit|parliament|minister|tories|labour|pope|church|police|racis\w*|"
                 r"immigr\w*|refugees?|capitalis\w*|corporat\w*|billionaires?|inequality|"
                 r"abortion|nra|supreme court|military|war|israel|palestin\w*|putin|"
                 r"kim jong|north korea)\b")
NONNEWS = pattern(r"\b(?:quiz|quizzes|listicle|word search|find out|they said what|how many|"
                  r"can you|which of|check out|here are|these \d+|"
                  r"\d+ (?:things|ways|reasons|tips|times|pictures|photos))\b|"
                  r"(?:^|:\s*)\d+\s+|^video:")
CLAIM = pattern(r"\b(?:is|are|was|were|has|have|had|will|can|cannot|could|became|becomes|"
                r"invented|invents?|discovered|discovers?|created|creates?|built|builds|"
                r"founded|founds|caused|causes|killed|kills|banned|bans|declared|declares|"
                r"announced|announces|proved|proves|revealed|reveals|admitted|admits|died|dies|"
                r"won|wins|defeated|defeats|signed|signs|ordered|orders|arrested|arrests|"
                r"launched|launches|stole|steals|stolen|sold|sells|bought|buys|outlawed|"
                r"outlaws|renamed|renames|destroyed|destroys|increased|increases|decreased|"
                r"decreases|prevented|prevents|produces|produced|contains|contained|means|"
                r"meant|requires|required|lives|lived|eats|ate|makes|made|weighs|weighed)\b")
CAPTION = pattern(r"\b(?:footage|sketch|depict\w*|shown|showing|looking at|standing|sitting|"
                  r"seen here|on the left|on the right|poster|postcard|postal stamps?|"
                  r"photograph\w*|concept art|illustrat\w*|costume|cosplay)\b")
CORRECTION = pattern(r"\b(?:correction|debunk\w*|not true|actually false|fact.check|"
                     r"fake news|clickbait)\b|\|")
NONASSERTION = pattern(r"^(?:how|why|what|when|where|who|which|should|would|could|can|do|"
                       r"does|did|is|are|was|were|has|have|i|my|we|our|you|your|this|"
                       r"these|that|those|here|please|anyone|meta|discussion)\b")
INVENTION_CUE = pattern(r"\b(?:invent\w*|discover\w*|creat\w*|clone\w*|alien\w*|"
                        r"time.travel\w*|secret\w*|actually|reveals?|revealed|experiment\w*|"
                        r"first.ever|world.s first|hybrid\w*|immortal\w*|teleport\w*|"
                        r"robot\w*|dinosaur\w*|extraterrestrial\w*|conspir\w*|cure\w*|hoax\w*)\b")
FANTASY = pattern(r"\b(?:hogwarts|harry potter|voldemort|mordor|gandalf|sauron|pokemon|"
                  r"pokémon|robotnik|sonic|minecraft|skyrim|mojave wasteland|ncr ranger|"
                  r"darth vader|jedi|star wars|spongebob|simpsons|fortnite|zelda)\b")
# Conservative exclusions for recognizable real-event captions. This is NOT
# an exhaustive fact-checker; unrecognized genuine claims can still survive.
ORDINARY_HISTORY = pattern(
    r"(?:kennedy|jfk).*(?:assassinat|dallas)|titanic.*(?:sank|sink)|"
    r"trump.*(?:syria|inaugurat|elected)|pelosi.*impeach|assange.*arrest|"
    r"(?:hitler|nazi|ss agent).*(?:jew|holocaust)|berlin wall.*(?:fall|built)|"
    r"(?:apollo|armstrong).*moon|world war.*(?:begin|end)|"
    r"golden spike|transcontinental railroad|declaration of independence|"
    r"pluto.*(?:discover|nam)|first shark fossile|"
    r"ligo|gravitational waves|tyson foods|chicken mcnugget|steve shubin|"
    r"nasa.*contact.*alien|sickle cell.*herrick|"
    r"mark zuckerberg is actually a human")
PERSUASION = pattern(r"\b(?:join|fight|defend|unite|resist|enlist|obey|boycott|vote|"
                     r"protect|liberate|destroy|crush|support|stop|remember|beware|"
                     r"we must|you must|our enemy|our enemies|workers of|death to|"
                     r"long live|forward to|victory to|stand up)\b")
SLOGAN_START = pattern(r"^[\s\"'«“‘]*(?:join|fight|defend|unite|resist|enlist|obey|"
                       r"boycott|vote|protect|liberate|destroy|crush|support|stop|"
                       r"remember|beware|workers|long live|death to|we must|you must)\b")
NEWS_START = pattern(r"^(?:report\b|study\b|new study\b|scientists?\b|researchers?\b|"
                     r"experts?\b|doctors?\b|news:|breaking:)")


def normalize(value: str) -> str:
    return " ".join(value.split())


def group_key(text: str) -> str:
    return normalize(text).casefold()


def readable(text: str) -> bool:
    return (6 <= len(WORDS.findall(text)) <= 80
            and sum(char.isalpha() for char in text) >= 25
            and not META.search(text) and not URL.search(text))


def read_rows(path: Path, delimiter: str, required: set[str]):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = reader.fieldnames or []
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError(f"Missing/duplicate columns in {path}")
        for index, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed record {index} in {path}")
            yield index, row


def recovery_key(row: dict) -> tuple:
    return (row["source"], normalize(row["text"]), row["original_label"],
            row["original_subtype"], row["source_split"])


def positive_combined_rows(audit: Counter) -> list[dict]:
    """Use the existing combined positives; recover genuine raw provenance."""
    selected = {}
    required = {"text", "source", "original_label", "original_subtype", "source_split"}
    for _, row in read_rows(COMBINED, ",", required):
        audit[f"combined.loaded.{row['source']}"] += 1
        if row["source"] == "fakeddit":
            continue  # Full raw TSVs replace the 30,000-record subsample.
        expected = {"native_ads": "1", "propaganda": "1", "manipulation": "true"}
        if row["original_label"] != expected.get(row["source"]):
            audit[f"combined.rejected_nonpositive.{row['source']}"] += 1
            continue
        selected[recovery_key(row)] = row
    recovered = {}

    def recover(row, raw, path, number, original_id=""):
        key = recovery_key(row)
        if key in selected and key not in recovered:
            recovered[key] = {**selected[key], "original_id": original_id,
                              "original_file": str(path.relative_to(ROOT)),
                              "original_row_number": number,
                              "service": raw.get("service", ""), "query": raw.get("query", "")}
            if row["source"] == "native_ads":
                recovered[key]["text"] = advertising_sentence(raw)[0]

    path = ROOT / "data/raw/Manipulation/manipulational_conversation.jsonl"
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            row = {"source": "manipulation", "source_split": "",
                   "text": "\n".join(m["text"] or "" for m in raw["messages"]),
                   "original_label": json.dumps(raw["is_manipulation"]),
                   "original_subtype": raw["manipulation_type"]}
            recover(row, raw, path, number, raw["conversation_id"])
    for split in ["train", "validation", "test"]:
        paths = sorted((ROOT / "data/raw/Native-Ads" / split).rglob("*.arrow"))
        if not paths:
            raise FileNotFoundError(f"Missing Native-Ads Arrow files for {split}")
        for path in paths:
            with pa.memory_map(str(path), "r") as handle:
                number = 0
                for batch in pa.ipc.open_stream(handle):
                    for raw in batch.to_pylist():
                        number += 1
                        subtype = {k: raw.get(k) for k in ["advertisement", "meta_topic"]
                                   if raw.get(k) is not None and raw.get(k) != ""}
                        row = {"source": "native_ads", "source_split": split,
                               "text": raw["response"], "original_label": str(raw["label"]),
                               "original_subtype": json.dumps(subtype, ensure_ascii=False, sort_keys=True) if subtype else ""}
                        recover(row, raw, path, number, raw["id"])
    path = ROOT / "data/raw/propaganda_dataset/propaganda_dataset.csv"
    for number, raw in read_rows(path, ",", {"text", "label"}):
        recover({"source": "propaganda", "source_split": "", "text": raw["text"],
                 "original_label": raw["label"], "original_subtype": ""}, raw, path, number)
    if recovered.keys() != selected.keys():
        raise ValueError(f"Unmatched combined positives: {len(selected.keys() - recovered.keys())}")
    return [recovered[key] for key in selected]


def classify_fakeddit(row: dict, audit: Counter):
    """Return a heuristic category, priority, and basis, or reject the row."""
    sub, text, label = row["subreddit"], normalize(row["title"]), row["6_way_label"]

    def reject(reason):
        audit[f"fakeddit.rejected.{sub}.{reason}"] += 1
        return None

    if not text:
        return reject("empty_text")
    if not readable(text):
        return reject("readability_length_url_or_discussion")
    if sub in SATIRE_SOURCES and label == "1":
        domain = row["domain"].lower()
        clickhole = domain == "clickhole.com" or domain.endswith(".clickhole.com")
        parody = (not PUBLIC.search(text) and not VISUAL.search(text)
                  and not NONNEWS.search(text) and "?" not in text
                  and (NEWS.search(text) or clickhole))
        if parody:
            # Extra exclusions can only narrow the previously identified pool.
            if CAPTION.search(text) or META.search(text):
                return reject("parody_visual_or_discussion")
            priority = 3 if NEWS_START.search(text) else 2 if NEWS.search(text) else 1
            return 5, priority, "Satire-source news imitation / ClickHole headline heuristic"
        if VISUAL.search(text) or NONNEWS.search(text) or CAPTION.search(text) or "?" in text:
            return reject("satire_visual_quiz_question_or_caption")
        priority = 3 if PUBLIC.search(text) else 1
        return 2, priority, "Satire-source headline; public/social-affairs indicators prioritized"
    if sub == "propagandaposters" and label == "5" and row["3_way_label"] == "1":
        if VISUAL.search(text) or CAPTION.search(text) or CORRECTION.search(text) or "?" in text:
            return reject("poster_description_visual_or_question")
        if not PERSUASION.search(text):
            return reject("no_self_contained_persuasive_language")
        priority = 3 if SLOGAN_START.search(text) else 1
        return 3, priority, "Standalone persuasive slogan heuristic; not inferred from image flag"
    if sub in {"fakefacts", "fakehistoryporn"}:
        if (sub == "fakefacts" and label != "5") or (sub == "fakehistoryporn" and label != "2"):
            return reject("unexpected_original_label")
        if "?" in text or NONASSERTION.search(text) or not CLAIM.search(text):
            return reject("not_a_standalone_assertion")
        if VISUAL.search(text) or CAPTION.search(text):
            return reject("visual_caption_dependency")
        if CORRECTION.search(text):
            return reject("correction_or_debunk")
        if FANTASY.search(text):
            return reject("explicit_fictional_universe")
        if ORDINARY_HISTORY.search(text):
            return reject("recognizable_factual_event_or_statement")
        if sub == "fakehistoryporn" and not INVENTION_CUE.search(text):
            return reject("historical_caption_without_invention_indicator")
        return (6, 3 if sub == "fakefacts" else 1,
                "Claim-like text plus fakefacts/invention-cue provenance; falsity NOT verified")
    return reject("unexpected_provenance")


def gather_candidates(audit: Counter) -> dict[str, dict]:
    candidates = {}
    conflicts = set()

    def add(raw: dict, label: int, priority: int, basis: str):
        record = dict.fromkeys(FIELDS, "")
        record.update({k: v for k, v in raw.items() if k in record})
        text = normalize(record["text"])
        if not text:
            audit["candidate.rejected_empty"] += 1
            return
        key = group_key(text)
        record.update(text=text, final_label=label, final_category=CATEGORIES[label],
                      selection_basis=basis, heuristic_priority=priority,
                      text_group_id=hashlib.sha256(key.encode("utf-8")).hexdigest())
        record["_splits"] = {record["source_split"]} - {""}
        record["_ids"] = {record["original_id"]} - {""}
        audit[f"candidate.accepted_before_dedup.{label}"] += 1
        if key in conflicts:
            audit["dedup.further_conflicting_records_removed"] += 1
            return
        previous = candidates.get(key)
        if previous:
            kind = "exact_normalized" if previous["text"] == text else "case_equivalent"
            audit[f"dedup.{kind}_repeated_records"] += 1
            if previous["final_label"] != label:
                if {previous["final_label"], label} == {2, 5}:
                    audit["dedup.satire_parody_conflicts_resolved_to_parody"] += 1
                    if label == 5:
                        record["_splits"].update(previous["_splits"])
                        record["_ids"].update(previous["_ids"])
                        candidates[key] = record
                    else:
                        previous["_splits"].update(record["_splits"])
                        previous["_ids"].update(record["_ids"])
                    return
                del candidates[key]
                conflicts.add(key)
                audit["dedup.cross_category_conflict_groups_excluded"] += 1
                return
            previous["_splits"].update(record["_splits"])
            previous["_ids"].update(record["_ids"])
            return
        candidates[key] = record

    for row in positive_combined_rows(audit):
        label = {"native_ads": 1, "propaganda": 3, "manipulation": 4}[row["source"]]
        add(row, label, 10, NATIVE_ADS_BASIS if label == 1 else "Approved positive original source label")
    required = {"title", "subreddit", "domain", "id", "2_way_label", "3_way_label", "6_way_label"}
    for filename, split in [("all_train.tsv", "train"), ("all_test_public.tsv", "test")]:
        path = ROOT / "data" / filename
        for number, raw in read_rows(path, "\t", required):
            audit[f"fakeddit.loaded.{split}"] += 1
            if raw["subreddit"] not in WANTED_SOURCES:
                audit["fakeddit.rejected.non_target_subreddit"] += 1
                continue
            decision = classify_fakeddit(raw, audit)
            if decision is None:
                continue
            label, priority, basis = decision
            row = {**raw, "text": raw["title"], "source": "fakeddit",
                   "original_label": raw["6_way_label"], "source_split": split,
                   "original_id": raw["id"], "original_title": raw["title"],
                   "original_file": str(path.relative_to(ROOT)), "original_row_number": number,
                   "label_2way": raw["2_way_label"], "label_3way": raw["3_way_label"],
                   "label_6way": raw["6_way_label"]}
            add(row, label, priority, basis)
    for row in candidates.values():
        row["observed_source_splits"] = json.dumps(sorted(row.pop("_splits")))
        row["matching_original_ids"] = json.dumps(sorted(row.pop("_ids")))
    return candidates


def choose(candidates: dict, audit: Counter) -> tuple[list[dict], dict[int, int]]:
    """Sample deterministically, balancing manipulation across its subtypes."""
    rng = random.Random(RANDOM_STATE)
    pools = {label: [] for label in CATEGORIES}
    for row in candidates.values():
        pools[row["final_label"]].append(row)
    available = {label: len(rows) for label, rows in pools.items()}
    result = []
    for label, rows in pools.items():
        if label == 4:
            groups = defaultdict(list)
            for row in rows:
                groups[row["original_subtype"]].append(row)
            quotas = dict.fromkeys(sorted(groups), 0)
            remaining = min(TARGET, len(rows))
            while remaining:
                for subtype in quotas:
                    if quotas[subtype] < len(groups[subtype]):
                        quotas[subtype] += 1
                        remaining -= 1
                        if not remaining:
                            break
            selected = []
            for subtype, quota in quotas.items():
                selected.extend(rng.sample(groups[subtype], quota))
        else:
            shuffled = list(rows)
            rng.shuffle(shuffled)
            # Source propaganda has priority 10: all its suitable positives
            # precede supplemental slogans. Random ties use only seed 42.
            shuffled.sort(key=lambda row: -row["heuristic_priority"])
            selected = shuffled[:TARGET]
        result.extend(selected)
        audit[f"selection.not_selected_due_to_target.{label}"] = len(rows) - len(selected)
    rng.shuffle(result)
    return result, available


def validate(rows: list[dict]) -> dict:
    """Check mandatory schema, category correspondence, and global uniqueness."""
    if not rows:
        raise ValueError("Empty final dataset")
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError("Unexpected output columns")
        label = int(row["final_label"])
        if label not in CATEGORIES or row["final_category"] != CATEGORIES[label]:
            raise ValueError("Invalid label/category mapping")
        if not row["text"].strip() or normalize(row["text"]) != row["text"]:
            raise ValueError("Empty or non-normalized output text")
    keys = [group_key(row["text"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Case/whitespace-equivalent duplicates remain")
    labels = Counter(int(row["final_label"]) for row in rows)
    if set(labels) != set(CATEGORIES):
        raise ValueError("Output must contain every label from 1 through 6")
    if any(count > TARGET for count in labels.values()):
        raise ValueError("Category exceeds target")
    return {"records": len(rows), "category_counts": dict(sorted(labels.items())),
            "empty_text": 0, "duplicate_normalized_text": 0,
            "duplicate_casefolded_text": 0, "cross_category_text_overlap": 0,
            "labels_exactly_1_through_6": True, "category_names_match": True}


def make_report(rows, available, audit, validation) -> str:
    lines = ["FINAL SIX-CATEGORY DATASET REPORT", "",
             "Construction: local source labels and explicit text heuristics; no APIs or generated data.",
             f"Random state: {RANDOM_STATE}; target per category: {TARGET}",
             "NOT manually verified ground truth. Shortages are retained, never padded.", "",
             "CATEGORY COUNTS (eligible after dedup; selected; shortage)"]
    counts = Counter(int(row["final_label"]) for row in rows)
    for label, name in CATEGORIES.items():
        lines.append(f"{label} {name}: eligible={available[label]}, selected={counts[label]}, shortage={TARGET-counts[label]}")
    lines.extend([f"Total records: {len(rows)}", f"Balanced at 1500 each: {all(counts[i] == TARGET for i in CATEGORIES)}", ""])

    def distribution(title, values):
        lines.append(title)
        for key, count in sorted(Counter(values).items()):
            lines.append(f"  {key}: {count}")
        lines.append("")

    distribution("SOURCE DISTRIBUTION", (r["source"] for r in rows))
    distribution("CATEGORY / SOURCE DISTRIBUTION", (f"{r['final_label']} / {r['source']}" for r in rows))
    distribution("MANIPULATION SUBTYPES", (r["original_subtype"] for r in rows if int(r["final_label"]) == 4))
    for label in [2, 3, 5, 6]:
        distribution(f"FAKEDDIT SUBREDDITS: {label} {CATEGORIES[label]}",
                     (r["subreddit"] for r in rows if r["source"] == "fakeddit" and int(r["final_label"]) == label))
    distribution("ORIGINAL SOURCE / SPLIT", (f"{r['source']} / {r['source_split'] or '(unsplit)'}" for r in rows))
    lines.extend(["FILTERING, DUPLICATE REMOVALS, AND SAMPLING AUDIT",
                  "Rejection counts refer to input rows; eligible pools refer to unique texts.",
                  "Repeated eligible texts are counted as exact-normalized or additional case-equivalent duplicates.",
                  "Conflicting non-satire/parody categories exclude both the stored representative and the repeat."])
    lines.extend(f"  {key}: {value}" for key, value in sorted(audit.items()))
    lines.extend(["", "VALIDATION", json.dumps(validation, indent=2),
                  "No normalized text occurs in multiple categories.", "",
                  "SELECTION RULES",
                  "1: combined native_ads original_label=1; seeded sampling.",
                  "2: full Fakeddit satire sources; public/social-affairs keyword matches first; exclude parody pool.",
                  "3: all available combined propaganda positives first; then persuasive standalone slogan candidates.",
                  "4: combined manipulation true; equal subtype quotas when available.",
                  "5: subset of previously identified news-pattern/ClickHole parody pool; news-format matches first.",
                  "6: fakefacts assertions and fakehistoryporn assertions with invention-related cues; no quota padding.",
                  "Image presence alone is never an exclusion; explicit visual/caption dependency is screened in text.",
                  "", "LIMITATIONS",
                  "Heuristic filtering is not manual verification; all six labels remain source-supervised or heuristic.",
                  "Satire and parody overlap conceptually; keyword/domain selection cannot establish author intent.",
                  "Fabrication screening does not verify falsity or establish presentation as genuine.",
                  "Jokes, genuine facts, subtle captions, and non-news assertions may survive Fabrication screening.",
                  "The recognizable-real-event exclusion list is conservative and incomplete, not a fact-checker.",
                  "Poster titles may quote persuasive slogans but can still require context or an image.",
                  "Readability gates are English-oriented and can reject useful short or non-English titles.",
                  "Source labels do not certify human authorship; no new synthetic text was generated.",
                  "No ordinary historical caption was used solely to fill the Fabrication quota.",
                  "Only exact and case/whitespace-equivalent duplicates are removed; semantic/template leakage remains possible.",
                  "Raw training and test records are pooled and original splits preserved; this is NOT an evaluation split.",
                  "Do not evaluate on overlapping original datasets. Group related claims/templates before a future split.",
                  "Matched IDs/splits describe accepted duplicate candidates; they are not an exhaustive related-record graph.",
                  "No model training or application changes were performed.", "",
                  f"CSV: {OUTPUT}", f"REPORT: {REPORT}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Compute report without writing any files")
    args = parser.parse_args()
    if not args.preview and (OUTPUT.exists() or REPORT.exists()):
        raise FileExistsError("Refusing to overwrite an existing final CSV or report")
    audit = Counter()
    print("Reading existing local positives and full Fakeddit TSVs...", flush=True)
    candidates = gather_candidates(audit)
    rows, available = choose(candidates, audit)
    validation = validate(rows)
    report = make_report(rows, available, audit, validation)
    if args.preview:
        print(report)
        return
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    # Validate the actual serialized CSV before reporting success.
    saved = [row for _, row in read_rows(OUTPUT, ",", set(FIELDS))]
    if validate(saved) != validation:
        raise ValueError("Saved CSV validation differs from in-memory validation")
    with REPORT.open("x", encoding="utf-8", newline="") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
