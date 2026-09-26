"""Offline split, leakage, label, and loss checks; never start model training."""

import csv
import json
import sys
import unittest
from unittest.mock import patch
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_distilbert_six_category as experiment

# The corrected run keeps the original run intact and uses the same test suite.
if (ROOT / "results/native_ads_correction/audit.json").exists():
    from train_distilbert_native_ads_corrected import configure_experiment
    configure_experiment()


class SixCategoryPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = experiment.load_rows()
        with experiment.SPLIT_FILE.open(encoding="utf-8", newline="") as handle:
            cls.manifest = list(csv.DictReader(handle))
        cls.splits = [row["split"] for row in cls.manifest]
        cls.plan = json.loads(experiment.PLAN_FILE.read_text(encoding="utf-8"))
        if experiment.sha256_file(experiment.SPLIT_FILE) != cls.plan["split_manifest_sha256"]:
            raise AssertionError("Frozen split manifest changed")
        cls.groups = [row["group_id"] for row in cls.manifest]
        # The frozen grouping is distributed; raw reconstruction is an additional
        # local check and must not make an ordinary partner checkout depend on raw data.
        if (ROOT / "data/raw/Manipulation/manipulational_conversation.jsonl").is_file():
            with patch.object(experiment, "base_snapshot", return_value=
                              ROOT / "models/distilbert_six_category_native_ads_corrected"):
                regenerated, _ = experiment.make_groups(cls.rows)
            if regenerated != cls.groups:
                raise AssertionError("Raw-source grouping differs from the frozen manifest")

    def test_dataset_unchanged_and_every_row_used_once(self):
        self.assertEqual(experiment.sha256_file(experiment.DATASET), self.plan["dataset_sha256"])
        self.assertEqual([int(row["dataset_row"]) for row in self.manifest], list(range(len(self.rows))))
        counts = experiment.verify_split(self.rows, self.groups, self.splits)
        self.assertEqual(counts, self.plan["split_counts"])
        self.assertEqual(self.plan["split_totals"], {"train": 5921, "validation": 1269, "test": 1269})

    def test_grouped_split_is_reproducible(self):
        if "preserved_split_manifest" in self.plan:
            source = ROOT / "results/distilbert_six_category/split_manifest.csv"
            self.assertEqual(experiment.sha256_file(source), self.plan["preserved_split_manifest_sha256"])
            with source.open(encoding="utf-8", newline="") as handle:
                self.assertEqual([row["split"] for row in csv.DictReader(handle)], self.splits)
            experiment.verify_split(self.rows, self.groups, self.splits)
            return
        labels = np.array([int(row["final_label"]) - 1 for row in self.rows])
        self.assertEqual(experiment.grouped_stratified_split(labels, self.groups), self.splits)

    def test_known_template_and_query_families_do_not_cross(self):
        families = defaultdict(set)
        for row, split in zip(self.rows, self.splits):
            if row["source"] == "propaganda":
                families[("propaganda", experiment.propaganda_template(row["text"]))].add(split)
            if row["source"] == "native_ads":
                families[("query", experiment.lexical_text(row["query"]))].add(split)
        self.assertTrue(all(len(values) == 1 for values in families.values()))
        self.assertEqual(sum(key[0] == "propaganda" for key in families), 20)

    def test_group_crossing_is_rejected(self):
        members = defaultdict(list)
        for index, group in enumerate(self.groups):
            members[group].append(index)
        group = next(indices for indices in members.values() if len(indices) > 1)
        changed = list(self.splits)
        changed[group[0]] = "test" if changed[group[0]] != "test" else "train"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            experiment.verify_split(self.rows, self.groups, changed)

    def test_weights_come_only_from_training_labels(self):
        labels = [int(row["final_label"]) - 1 for row, split in zip(self.rows, self.splits) if split == "train"]
        weights = experiment.training_class_weights(labels)
        self.assertAlmostEqual(weights[0], 5921 / (6 * 1050))
        self.assertAlmostEqual(weights[5], 5921 / (6 * 671))
        self.assertAlmostEqual(np.mean([weights[label] for label in labels]), 1.0)

    def test_weighted_loss_accumulation_and_unweighted_evaluation(self):
        import torch
        from torch.nn.functional import cross_entropy
        logits = torch.tensor([[1., 0., 0., 0., 0., 2.], [0., 1., 2., 0., 0., 1.],
                               [1., 0., 1., 0., 0., 2.], [2., 1., 0., 0., 0., 1.]])
        labels = torch.tensor([0, 5, 5, 0])
        weights = list(self.plan["training_only_class_weights"].values())
        whole = experiment.classification_loss(logits, labels, weights, True)
        chunks = sum(experiment.classification_loss(logits[i:i+2], labels[i:i+2], weights, True, 4)
                     for i in [0, 2])
        self.assertTrue(torch.allclose(whole, chunks))
        evaluation = experiment.classification_loss(logits, labels, weights, False)
        self.assertTrue(torch.allclose(evaluation, cross_entropy(logits, labels)))
        self.assertFalse(logits.requires_grad)

    def test_metadata_never_enters_model_inputs(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(ROOT / "models/distilbert_six_category_native_ads_corrected"), local_files_only=True)
        records = [dict(row) for row in self.rows[:2]]
        before = experiment.model_dataset(records, [0, 1], tokenizer).to_dict()
        for row in records:
            for key in set(row) - {"text", "final_label"}:
                row[key] = "metadata must not influence inputs"
        after = experiment.model_dataset(records, [0, 1], tokenizer).to_dict()
        self.assertEqual(before, after)
        self.assertEqual(set(before), {"input_ids", "attention_mask", "labels"})
        self.assertEqual(before["labels"], [int(row["final_label"]) - 1 for row in records])

    def test_metric_names_and_project_mapping(self):
        metrics = experiment.metric_values(np.arange(6), np.arange(6))
        self.assertEqual(set(metrics), {"accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"})
        self.assertTrue(all(value == 1.0 for value in metrics.values()))
        mapping = experiment.label_mapping()
        for project_id, name in experiment.CATEGORIES.items():
            self.assertEqual(mapping["internal_id_to_category"][str(project_id - 1)], name)
            self.assertEqual(mapping["internal_id_to_project_id"][str(project_id - 1)], project_id)


if __name__ == "__main__":
    unittest.main()
