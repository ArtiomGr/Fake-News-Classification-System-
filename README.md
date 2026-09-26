# Fake News & Content Classification System

**Six-Category Text Classification and Sentiment Analysis**

The final university-project dashboard compares **BERT**, **corrected DistilBERT**, and **TF-IDF + Logistic Regression** on the same input. **VADER** analyzes emotional tone separately. All application inference is local; no API, downloads or training occur when running the app.

## Partner setup after cloning or pulling

Use **Python 3.13**, matching the tested environment. Open PowerShell in the repository root. After cloning your shared repository (or running `git pull` in an existing checkout):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements pin the tested package versions, including scikit-learn/joblib compatibility with the saved LR pipeline. Package installation requires internet access; subsequent inference is offline. A CPU is sufficient. The complete models need about 710 MB of artifact storage, plus several GB for Python dependencies and working RAM.

**A Git clone alone does not include the two Transformer weight files. Complete the next step before Analyze or model-dependent tests. Do not retrain.**

## Obtain the exact final model weights

| Model artifact | Exact bytes | MiB | Delivery |
|---|---:|---:|---|
| BERT `model.safetensors` | 437,970,928 | 417.68 | Separate handoff archive |
| Corrected DistilBERT `model.safetensors` | 267,844,872 | 255.44 | Separate handoff archive |
| LR `pipeline.joblib` | 1,255,900 | 1.20 | Included in Git |

GitHub blocks files larger than **100 MiB** in ordinary Git. Git LFS or release assets are alternatives; this handoff uses a separate archive so no LFS account/quota is required. [GitHub file-limit documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

**Project owner:** the ready-to-share archive is `handoff/final-transformer-weights.zip` (705,816,208 bytes). It contains only the two exact evaluated weight files, with their destination directories. Send it directly to your partner using your shared storage/USB/file-transfer method. Alternatively, attach it to a GitHub Release after separately approving publication. Nothing has been uploaded, and there is no hosted download URL yet. The archive and original local models are preserved and ignored by Git.

**Project partner:** obtain that archive from the project owner, then extract it into the cloned repository root:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\final-transformer-weights.zip" -DestinationPath .
.\.venv\Scripts\python.exe -B src/verify_artifacts.py
```

For an existing checkout, inspect any existing weights before replacing them; `Expand-Archive` deliberately does not overwrite files by default. You can instead copy the two weight files manually into the directories below. Do not use the old Fakeddit models or uncorrected DistilBERT.

```text
models/
  artifact_manifest.json
  bert_six_category_final/
    model.safetensors          # obtained separately
    config.json
    experiment.json
    label_mapping.json
    tokenizer.json
    tokenizer_config.json
  distilbert_six_category_native_ads_corrected/
    model.safetensors          # obtained separately
    config.json
    label_mapping.json
    tokenizer.json
    tokenizer_config.json
  logistic_regression_six_category_final/
    pipeline.joblib
    experiment.json
    label_mapping.json
```

Everything above except the two weight files is included in the proposed Git changes. `verify_artifacts.py` checks all file sizes and SHA-256 hashes against `models/artifact_manifest.json`, without importing ML libraries. Load the joblib artifact only from this trusted project handoff.

## Run the final app

```powershell
.\.venv\Scripts\python.exe -B -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8501**. Paste text, select **Analyze**, and review the three predictions, agreement, probability tabs, and VADER. **Clear** resets input/results; editing input removes stale results. Stop and restart Streamlit after changing Python inference code. Missing/mismatched models produce an error rather than silently using legacy models.

Final labels, ordered by project ID:

1. Native Advertising
2. News Satire
3. Propaganda
4. Manipulation
5. News Parody
6. Fabrication

Saved label mappings translate internal IDs 0–5 to project IDs 1–6. Short Transformer input uses direct evaluation-mode inference. Longer input uses complete 512-token windows including special tokens, 16-token overlap, and arithmetic-mean window probabilities. LR processes full text; VADER is independent. Input limit: 50,000 characters. Pasted app text is not saved to disk.

## Dataset, metrics and provenance

The **final** 8,459-record dataset is included at `data/processed/final_six_category_dataset.csv` (4,400,900 bytes), together with the original construction report. The later correction is documented in `results/distilbert_six_category_native_ads_corrected/correction_report.md`. This is the frozen corrected dataset, not an intermediate/raw dataset. It supports the split/integrity tests and final comparison; it is not needed for ordinary app inference. All three models share exactly **5,921 train / 1,269 validation / 1,269 test** rows.

| Model | Test accuracy | Test macro F1 |
|---|---:|---:|
| BERT | 97.48% | 97.42% |
| Corrected DistilBERT | 97.87% | 97.77% |
| TF-IDF + Logistic Regression | 84.16% | 82.80% |

Final evaluation reports, per-class results, confusion matrices, row-level predictions, split manifests, label mappings and training plans are included under the three final `results/<model>/` directories. `results/final_model_comparison/` contains the shared comparison and evaluated-weight registry. Historical absolute paths inside experiment records document the original run; application paths are resolved relative to the clone.

The old pre-correction manifest/plan and annotation provenance are retained to explain the correction. Existing tracked top-level legacy metrics and legacy training scripts are historical work, not the final comparison. Do not run training/preprocessing scripts to set up the application. The optional historical API-labeling pilot is also not part of the app or its required dependencies.

See [the final system report](results/final_system_audit/final_report.md) for the model audit, checkpoint-export repair, metrics and limitations, and [the Git review](docs/GIT_REVIEW.md) for what is included/excluded. `.gitattributes` preserves exact artifact bytes across checkouts so line-ending conversion does not invalidate hashes.

## Offline verification

After installing dependencies and obtaining the two weights:

```powershell
.\.venv\Scripts\python.exe -B src/verify_artifacts.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Tests cover all classifiers, mappings, direct/module/Streamlit parity, chunk coverage, Analyze/Clear, agreement and independent VADER. The raw Native-Ads traceability audit skips when intentionally excluded raw source/backup files are absent. All other tests work with the handoff; no original Hugging Face cache or training checkpoints are needed.

Optional diagnostic commands (write refreshed local evaluation outputs; never train):

```powershell
.\.venv\Scripts\python.exe -B src/audit_final_parity.py
.\.venv\Scripts\python.exe -B src/final_test.py
```

TravelPro reproduces **Native Advertising — 0.6786684393882751** across direct corrected DistilBERT, `src/predict.py`, and Streamlit. The 12 fixed probes match 10/12 intended categories for BERT/DistilBERT and 6/12 for LR; these probes are not an independent accuracy benchmark.

## Intentionally excluded from Git

- Raw datasets, intermediate merged data, and the pre-correction CSV backup.
- Transformer weights and the local handoff archive (shared separately as above).
- Training checkpoints, optimizer/RNG states and training-argument binaries.
- Training/runtime logs, caches, virtual environments and local secret files.
- Duplicate model-repair exports, superseded evaluation outputs and transient audit diagnostics.

Ignore rules do not delete these files locally. Required application/source/test files, final model metadata/tokenizers, LR, final dataset and final evaluation evidence remain eligible for Git.

Model confidence and agreement do not verify factual truth. Source/heuristic labels, source/style cues, overlapping satire/parody concepts and fewer Fabrication records limit generalization. Earlier test/sanity results informed the Native Advertising correction, so the recorded scores are regression evaluation rather than an untouched independent test. Long-document averaging is not a separately validated document-level benchmark.
