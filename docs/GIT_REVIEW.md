# Git handoff review

Prepared for review only: **8 modified tracked files, 96 untracked files, zero staged files**. Nothing committed, pushed or uploaded. No existing local files were deleted. The current Streamlit design was preserved.

## What the proposed commit includes

- Final Streamlit app, every source/test Python file, all three classifier integrations, VADER, README and pinned requirements.
- Production model configs, tokenizers, experiment metadata and six-category mappings; the complete 1,255,900-byte LR pipeline.
- Final corrected 8,459-record CSV (4,400,900 bytes) and original construction report. Raw/intermediate datasets are excluded.
- Final BERT, corrected DistilBERT and LR validation/test metrics, per-class reports, confusion matrices, predictions and frozen split manifests.
- Shared comparison metrics, evaluated-weight registry, fixed qualitative predictions, correction provenance and final project report.
- Only the old pre-correction split manifest/training plan, needed as provenance for the correction.
- `.gitattributes` to preserve artifact hashes and `src/verify_artifacts.py` plus `models/artifact_manifest.json` for partner verification.

Existing already-tracked legacy scripts/top-level evaluation files are retained as historical work; they are not new final results. Required code is not ignored. New experimental source scripts are retained for project history but are not part of app startup; the historical API pilot is optional and not in the required dependencies.

The complete proposed checkout has 130 files and approximately 15.77 MB before Git compression. No eligible file exceeds GitHub's 100 MiB ordinary-file limit.

## What remains local-only

Raw data, merged/intermediate data, pre-correction CSV backup, old model directories, Transformer weights, checkpoints, optimizer/RNG states, training/runtime logs, caches, environment/secret files, duplicate BERT repair outputs, transient runtime diagnostics and the handoff folder. Ignore rules do not delete these files.

- BERT weights: **437,970,928 bytes / 417.68 MiB**.
- Corrected DistilBERT weights: **267,844,872 bytes / 255.44 MiB**.
- LR pipeline: **1,255,900 bytes / 1.20 MiB**, included in Git.

The Transformers exceed [GitHub's ordinary 100 MiB limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Prepared separate archive: `handoff/final-transformer-weights.zip`, **705,816,208 bytes**, containing only their two weight files with repository-relative paths. Archive entries and SHA-256 values were verified against the manifest. Share this existing archive directly with the partner; no download is hosted yet. A future GitHub Release or Git LFS is optional, not configured or published in this task. README gives extraction, artifact verification, installation and launch commands. A clone alone cannot run all classifiers until these two files are supplied.

## Verification

- Original local checkout: **40 tests passed**, no skips.
- Simulated partner checkout: **40 tests run; 39 passed, 1 documented skip** (raw Native-Ads traceability needs excluded raw/backup files). All app/model/split checks passed.
- Simulation copied only Git-eligible files, then supplied the two final weights; it used a different repository directory and the existing tested Python environment. This was not a fresh pip installation.
- Removed test dependence on the original user path, external Hugging Face cache and training checkpoints. Split tests validate the saved manifest hash and group separation; raw grouping is additionally reconstructed when local raw data exists.
- Adapted the UI title check to the current HTML heading without changing the app design.
- All 14 required model files verified by size/hash; no model or dataset was retrained or changed.
- `pip check`: no broken requirements. Python syntax checks passed. All source/test/app Python files were checked against Git inclusion rules.
- Test subprocesses exited successfully. Python still reports the preexisting Windows temporary-directory cleanup permission warning at interpreter exit; this does not fail test assertions.

Local test transcripts and the simulated checkout remain under ignored `handoff/`. Nothing was removed after verification.

## Final git status (short)

```text
 M .gitignore
 M README.md
 M app/app.py
 M requirements.txt
 M src/compare_models.py
 M src/final_test.py
 M src/predict.py
 M tests/test_predict.py
?? .gitattributes
?? data/
?? docs/
?? models/
?? results/bert_six_category_final/
?? results/distilbert_six_category/
?? results/distilbert_six_category_native_ads_corrected/
?? results/final_model_comparison/
?? results/final_system_audit/
?? results/logistic_regression_six_category_final/
?? results/native_ads_correction/
?? src/audit_final_parity.py
?? src/build_final_six_category_dataset.py
?? src/build_reliable_dataset.py
?? src/correct_native_ads.py
?? src/evaluate_native_ads_correction.py
?? src/final_project.py
?? src/finalize_bert_checkpoint.py
?? src/llm_label_dataset.py
?? src/prepare_combined_dataset.py
?? src/train_distilbert_native_ads_corrected.py
?? src/train_distilbert_six_category.py
?? src/train_final_models.py
?? src/verify_artifacts.py
?? tests/test_app.py
?? tests/test_distilbert_six_category.py
?? tests/test_native_ads_correction.py
```

## Full file-by-file review

The following is `git status --short --untracked-files=all`; `??` files are eligible but not staged.

```text
 M .gitignore
 M README.md
 M app/app.py
 M requirements.txt
 M src/compare_models.py
 M src/final_test.py
 M src/predict.py
 M tests/test_predict.py
?? .gitattributes
?? data/processed/final_dataset_report.txt
?? data/processed/final_six_category_dataset.csv
?? docs/GIT_REVIEW.md
?? models/artifact_manifest.json
?? models/bert_six_category_final/config.json
?? models/bert_six_category_final/experiment.json
?? models/bert_six_category_final/label_mapping.json
?? models/bert_six_category_final/tokenizer.json
?? models/bert_six_category_final/tokenizer_config.json
?? models/distilbert_six_category_native_ads_corrected/config.json
?? models/distilbert_six_category_native_ads_corrected/label_mapping.json
?? models/distilbert_six_category_native_ads_corrected/tokenizer.json
?? models/distilbert_six_category_native_ads_corrected/tokenizer_config.json
?? models/logistic_regression_six_category_final/experiment.json
?? models/logistic_regression_six_category_final/label_mapping.json
?? models/logistic_regression_six_category_final/pipeline.joblib
?? results/bert_six_category_final/best_checkpoint.json
?? results/bert_six_category_final/label_mapping.json
?? results/bert_six_category_final/split_manifest.csv
?? results/bert_six_category_final/test_classification_report.json
?? results/bert_six_category_final/test_confusion_matrix.csv
?? results/bert_six_category_final/test_confusion_matrix.png
?? results/bert_six_category_final/test_metrics.json
?? results/bert_six_category_final/test_per_class_metrics.csv
?? results/bert_six_category_final/test_predictions.csv
?? results/bert_six_category_final/training_plan.json
?? results/bert_six_category_final/validation_classification_report.json
?? results/bert_six_category_final/validation_confusion_matrix.csv
?? results/bert_six_category_final/validation_confusion_matrix.png
?? results/bert_six_category_final/validation_metrics.json
?? results/bert_six_category_final/validation_per_class_metrics.csv
?? results/bert_six_category_final/validation_predictions.csv
?? results/distilbert_six_category/split_manifest.csv
?? results/distilbert_six_category/training_plan.json
?? results/distilbert_six_category_native_ads_corrected/best_checkpoint.json
?? results/distilbert_six_category_native_ads_corrected/correction_report.md
?? results/distilbert_six_category_native_ads_corrected/label_mapping.json
?? results/distilbert_six_category_native_ads_corrected/sanity_predictions.json
?? results/distilbert_six_category_native_ads_corrected/split_manifest.csv
?? results/distilbert_six_category_native_ads_corrected/test_classification_report.json
?? results/distilbert_six_category_native_ads_corrected/test_confusion_matrix.csv
?? results/distilbert_six_category_native_ads_corrected/test_confusion_matrix.png
?? results/distilbert_six_category_native_ads_corrected/test_metrics.json
?? results/distilbert_six_category_native_ads_corrected/test_per_class_metrics.csv
?? results/distilbert_six_category_native_ads_corrected/test_predictions.csv
?? results/distilbert_six_category_native_ads_corrected/training_plan.json
?? results/distilbert_six_category_native_ads_corrected/validation_classification_report.json
?? results/distilbert_six_category_native_ads_corrected/validation_confusion_matrix.csv
?? results/distilbert_six_category_native_ads_corrected/validation_confusion_matrix.png
?? results/distilbert_six_category_native_ads_corrected/validation_metrics.json
?? results/distilbert_six_category_native_ads_corrected/validation_per_class_metrics.csv
?? results/distilbert_six_category_native_ads_corrected/validation_predictions.csv
?? results/final_model_comparison/model_registry.json
?? results/final_model_comparison/representative_predictions.csv
?? results/final_model_comparison/representative_predictions.json
?? results/final_model_comparison/test_metrics.csv
?? results/final_model_comparison/validation_metrics.csv
?? results/final_system_audit/bert_checkpoint_restore/completed.json
?? results/final_system_audit/final_report.md
?? results/final_system_audit/travelpro_parity.json
?? results/logistic_regression_six_category_final/label_mapping.json
?? results/logistic_regression_six_category_final/split_manifest.csv
?? results/logistic_regression_six_category_final/test_classification_report.json
?? results/logistic_regression_six_category_final/test_confusion_matrix.csv
?? results/logistic_regression_six_category_final/test_confusion_matrix.png
?? results/logistic_regression_six_category_final/test_metrics.json
?? results/logistic_regression_six_category_final/test_per_class_metrics.csv
?? results/logistic_regression_six_category_final/test_predictions.csv
?? results/logistic_regression_six_category_final/training_plan.json
?? results/logistic_regression_six_category_final/training_summary.json
?? results/logistic_regression_six_category_final/validation_classification_report.json
?? results/logistic_regression_six_category_final/validation_confusion_matrix.csv
?? results/logistic_regression_six_category_final/validation_confusion_matrix.png
?? results/logistic_regression_six_category_final/validation_metrics.json
?? results/logistic_regression_six_category_final/validation_per_class_metrics.csv
?? results/logistic_regression_six_category_final/validation_predictions.csv
?? results/native_ads_correction/annotation_provenance.json
?? results/native_ads_correction/audit.json
?? results/native_ads_correction/pretraining_report.md
?? src/audit_final_parity.py
?? src/build_final_six_category_dataset.py
?? src/build_reliable_dataset.py
?? src/correct_native_ads.py
?? src/evaluate_native_ads_correction.py
?? src/final_project.py
?? src/finalize_bert_checkpoint.py
?? src/llm_label_dataset.py
?? src/prepare_combined_dataset.py
?? src/train_distilbert_native_ads_corrected.py
?? src/train_distilbert_six_category.py
?? src/train_final_models.py
?? src/verify_artifacts.py
?? tests/test_app.py
?? tests/test_distilbert_six_category.py
?? tests/test_native_ads_correction.py
```
