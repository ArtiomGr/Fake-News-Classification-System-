# Final local system report

Repository handoff note: this report records the original local finalization. Runtime logs, duplicate checkpoint-repair exports, diagnostic snapshots and `generated_files.txt` referenced below are intentionally local-only. See the current `README.md` and `docs/GIT_REVIEW.md` for the Git handoff, included artifacts, portable setup, and current verification results.

## Scope and final architecture

The application is **Fake News & Content Classification System**, subtitled **Six-Category Text Classification and Sentiment Analysis**. Its workflow is text input, three classifier cards, descriptive model agreement, probability tabs, separate VADER sentiment, and technical information. No GitHub push or deployment was performed. All inference is local.

Final project labels, in display order: **1 Native Advertising; 2 News Satire; 3 Propaganda; 4 Manipulation; 5 News Parody; 6 Fabrication.** Internal model IDs are 0–5; the saved mapping translates them to project IDs 1–6. No old model's outputs were renamed to create this task.

## Existing-model audit

| Artifact | Original task and split evidence | Saved old evaluation | Final-task compatibility |
|---|---|---|---|
| `BERT_Final/BERT_model/models/bert/` and cached `Artiomg1/truthlens-bert-base` | `src/train_bert.py` uses Fakeddit `clean_title`/`6_way_label`, stratified samples of 12,000 train / 3,000 validation / 5,000 test, seed 42, length 128, two epochs. `src/verify_bert.py` exposes the old six labels. | Accuracy 0.741; weighted F1 0.7328689458516412 | Incompatible; final-task BERT was trained once from locally cached original pretrained BERT. |
| `models/distilbert/` | Old Fakeddit task; source script uses the same 12,000 / 3,000 / 5,000 sampled sizes, length 128 and two epochs. | Accuracy 0.7276; weighted F1 0.7183230574031441 | Incompatible. |
| `models/distilbert_six_category/` | Six-category experiment before the annotated Native Advertising extraction correction. | Existing reports retained. | Superseded; not loaded in the final app. |
| `models/distilbert_six_category_native_ads_corrected/` | Corrected final dataset and saved frozen manifest. | Validation/test reports and checkpoints already complete. | Compatible; preserved without retraining or changes. |
| `models/baseline_model.pkl` and `models/tfidf_vectorizer.pkl` | `src/baseline.py` uses the original Fakeddit title/label split files. | Accuracy 0.6196422864312686; weighted F1 0.6466549498129381 | Incompatible; final-task TF-IDF/LR pipeline was trained once. |

Old artifacts have weaker embedded provenance; their task/sampling details above come from the associated repository scripts and saved reports, not an independently recorded frozen manifest. Legacy files were retained.

## Final model paths and shared experiment

Paths are relative to `C:\Users\97250\Desktop\FinPro`:

- BERT: `models/bert_six_category_final/`
- Corrected DistilBERT: `models/distilbert_six_category_native_ads_corrected/`
- TF-IDF + Logistic Regression: `models/logistic_regression_six_category_final/`

All use `data/processed/final_six_category_dataset.csv`: 8,459 records, 1,500 in each of the first five categories and 959 Fabrication records. Dataset SHA-256: `a44b118d36942a151b99c8907ef613f98a0ad34bc566d22f3419c2ee41a54842`.

All use exactly the existing corrected manifest in `results/distilbert_six_category_native_ads_corrected/split_manifest.csv`: **5,921 training / 1,269 validation / 1,269 test**, SHA-256 `5149b198301db87d72ec016b803da03683f99ec9b1e3bb449295fc0cd37d5e94`. Row identities, labels, split hashes and known-group separation are checked. Existing checks report zero normalized-text and known-related-group overlap. The dataset was not rebuilt or cleaned again; Native Advertising still uses the existing annotated advertising passages.

BERT completed three epochs / 1,113 optimization steps. Selection was by validation macro F1, choosing epoch 2 / checkpoint 742. It used the corrected DistilBERT configuration: length 512, learning rate 2e-5, weight decay 0.01, train batch 8, evaluation batch 16, gradient accumulation 2, linear scheduling with 112 warmup steps, train-only balanced class weights, seed 42, deterministic float32 training. Initialization was original locally cached `bert-base-uncased`, not an old fine-tuned classifier. No hyperparameter search occurred.

The installed Transformers Trainer had a **BERT checkpoint restoration defect**: its raw state-dictionary loader skipped 50 saved LayerNorm `gamma`/`beta` parameters because the runtime expected `weight`/`bias`. Tensor comparison and missing/unexpected-key messages in the training log confirmed this. The completed best checkpoint was loaded through `from_pretrained`, every tensor was checked, and the canonical BERT weights were replaced with that existing checkpoint's exact bytes. Validation/test were then reevaluated. No additional training took place. The initial export and reports are preserved under `bert_checkpoint_restore/before/`; final evidence is in `bert_checkpoint_restore/completed.json`. The training script now explicitly restores converted checkpoint parameters strictly before saving/evaluation.

The LR pipeline uses TF-IDF unigrams/bigrams, maximum 30,000 features, min_df 2, max_df 0.95, sublinear TF, and retained negation; balanced Logistic Regression uses C=1, lbfgs, max_iter=2,000, tol=1e-4, seed 42. Vocabulary/IDF were fitted only on training records. It converged in 44 iterations with 14,394 vocabulary entries. No tuning occurred.

## TravelPro discrepancy: evidence and fix

Exact tested input, without surrounding quote characters:

> This article is brought to you by TravelPro. Book your next vacation with our exclusive summer deals.

A long-lived server on port 8501 was running a stale imported `predict` module. A fresh connection produced `cannot import name 'MAX_INPUT_CHARACTERS' from 'predict'` even though the constant existed in the on-disk module. A separate fresh server on 8502 already returned the correct 67.87% result. Evidence is in `port8501_details.json` and `running_servers_diagnosis.json`.

**The stale-runtime defect is confirmed. The historical News Parody 94.43% output was not reproduced, so its exact numerical origin is not established.** Tested alternate checkpoints and quoted-input variants did not reproduce that value. We do not attribute it conclusively to a particular old checkpoint or a label-order bug.

Both old repository servers were replaced. Streamlit and command-line inference now share `src/predict.py`, which loads the corrected directory, validates saved labels, checks evaluated weight hashes, and caches by artifact identity/version. Short text uses its normal token IDs directly; there is no decode/re-encode or unnecessary chunking. Model details expose loaded paths and hashes. Restart the server after changing Python inference code.

Direct corrected inference, `predict_model`, and the Streamlit AppTest path produce the same **internal class 0 / project label 1 / Native Advertising**, confidence **0.6786684393882751**. All six probabilities agree within absolute tolerance 1e-6:

| Category | Direct = prediction module = Streamlit |
|---|---:|
| Native Advertising | 0.6786684393882751 |
| News Satire | 0.00916996132582426 |
| Propaganda | 0.04513326659798622 |
| Manipulation | 0.00574896764010191 |
| News Parody | 0.2440718710422516 |
| Fabrication | 0.017207534983754158 |

All three paths use both model and tokenizer from `C:\Users\97250\Desktop\FinPro\models\distilbert_six_category_native_ads_corrected`. Weights SHA-256: `b1883d55a06b89285e4ab64be1977627dec9202633b86bc79397fad9b505bb48`; tokenizer JSON SHA-256: `435667fab0c06c165b1283ecb422497c37124f2d6a35b2ac73dc876332fc9518`.

The input is 22 tokens including `[CLS]` and `[SEP]`, one window, no padding required and no truncation. The attention mask is all ones. Logits are `[2.3874480724334717, -1.9167516231536865, -0.32306501269340515, -2.383664608001709, 1.3647780418395996, -1.287337303161621]`. Config `id2label` and `label2id` agree with saved project mappings. Complete token IDs and path-by-path outputs are in `travelpro_parity.json`.

## Long text, sentiment and interpretation

Both Transformers reserve space for special tokens within their 512-token maximum. Longer inputs use tokenizer overflow windows with 16 content tokens of overlap; arithmetic-mean window probabilities give the document prediction. Tests reconstruct the original content-token sequence and check the 512/513 boundary. LR transforms the complete input. Inputs are limited to 50,000 characters. App chunking does not change dataset preprocessing.

VADER is independent and reports overall sentiment, compound, positive, neutral and negative scores. TravelPro has positive tone, compound 0.128, positive 0.086, neutral 0.914, negative 0.0. Sentiment failure does not change classifier outputs.

For the defense: confidence is uncalibrated model output, not factual verification. Agreement does not prove truth. Dataset labels are heuristic/source-supervised, known grouping cannot eliminate arbitrary semantic paraphrases, and source/style cues can remain. Authors/domains/subreddits are not held out wholesale. Fabrication has fewer records, and satire/parody meanings overlap. Prior test and sanity results informed the earlier Native Advertising correction, so this is regression evaluation rather than an untouched independent test. Long-document probability averaging has not been separately validated as a document-level benchmark. The fixed 12 examples are qualitative probes, not an independent accuracy estimate or training data.

## Local launch

From the project root:

```powershell
.\.venv\Scripts\python.exe -B -m streamlit run app/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

URL: **http://127.0.0.1:8501**. Required model artifacts must already exist locally. No API or model download is used.

## Files changed or created in this finalization

- `README.md`
- `app/app.py`
- `src/predict.py`
- `src/compare_models.py`
- `src/final_test.py`
- `src/final_project.py`
- `src/train_final_models.py`
- `src/finalize_bert_checkpoint.py`
- `src/audit_final_parity.py`
- `tests/test_predict.py`
- `tests/test_app.py`
- Generated BERT artifacts: `models/bert_six_category_final/`, `results/bert_six_category_final/`
- Generated LR artifacts: `models/logistic_regression_six_category_final/`, `results/logistic_regression_six_category_final/`
- Comparison metrics, registry and qualitative predictions: `results/final_model_comparison/`
- Integrity snapshots, inference diagnostics, checkpoint-restoration evidence, live-test results and this report: `results/final_system_audit/`

`generated_files.txt` lists every individual file in these output directories, including checkpoints, model/tokenizer files, plans, manifests, reports, matrices and predictions. Preexisting untracked dataset/preprocessing files are not counted as new changes from this finalization. The protected corrected DistilBERT and finalized dataset remain unchanged.

## Final evaluation metrics

Each partition contains 1,269 records. Values below are percentages; the JSON/CSV artifacts retain full precision.

### Validation

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| BERT | 98.502758% | 98.320469% | 98.467593% | 98.389529% | 98.505328% |
| DistilBERT | 98.581560% | 98.333939% | 98.541667% | 98.428837% | 98.589080% |
| Logistic Regression | 84.791174% | 87.902777% | 85.037037% | 83.340501% | 83.044841% |

### Test

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| BERT | 97.478329% | 97.425904% | 97.421296% | 97.418727% | 97.475597% |
| DistilBERT | 97.872340% | 97.759986% | 97.791667% | 97.773018% | 97.873821% |
| Logistic Regression | 84.160757% | 89.007499% | 84.486111% | 82.803364% | 82.580964% |

### BERT: per-class test results

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Native Advertising | 99.5575% | 100.0000% | 99.7783% | 225 |
| News Satire | 95.8904% | 93.3333% | 94.5946% | 225 |
| Propaganda | 98.6667% | 98.6667% | 98.6667% | 225 |
| Manipulation | 100.0000% | 100.0000% | 100.0000% | 225 |
| News Parody | 93.9130% | 96.0000% | 94.9451% | 225 |
| Fabrication | 96.5278% | 96.5278% | 96.5278% | 144 |

Test mistakes: **32/1269**. Matrix rows are true categories and columns predicted categories; IDs follow the final 1?6 order.

| True / predicted | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 225 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1 | 210 | 1 | 0 | 11 | 2 |
| 3 | 0 | 0 | 222 | 0 | 1 | 2 |
| 4 | 0 | 0 | 0 | 225 | 0 | 0 |
| 5 | 0 | 6 | 2 | 0 | 216 | 1 |
| 6 | 0 | 3 | 0 | 0 | 2 | 139 |

Most confused category pairs (both directions combined): News Satire / News Parody: 17; News Satire / Fabrication: 5; Propaganda / News Parody: 3.

### DistilBERT: per-class test results

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Native Advertising | 99.5556% | 99.5556% | 99.5556% | 225 |
| News Satire | 95.1965% | 96.8889% | 96.0352% | 225 |
| Propaganda | 98.6607% | 98.2222% | 98.4410% | 225 |
| Manipulation | 100.0000% | 100.0000% | 100.0000% | 225 |
| News Parody | 97.2851% | 95.5556% | 96.4126% | 225 |
| Fabrication | 95.8621% | 96.5278% | 96.1938% | 144 |

Test mistakes: **27/1269**. Matrix rows are true categories and columns predicted categories; IDs follow the final 1?6 order.

| True / predicted | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 224 | 0 | 0 | 0 | 0 | 1 |
| 2 | 1 | 218 | 1 | 0 | 4 | 1 |
| 3 | 0 | 0 | 221 | 0 | 1 | 3 |
| 4 | 0 | 0 | 0 | 225 | 0 | 0 |
| 5 | 0 | 8 | 1 | 0 | 215 | 1 |
| 6 | 0 | 3 | 1 | 0 | 1 | 139 |

Most confused category pairs (both directions combined): News Satire / News Parody: 12; Propaganda / Fabrication: 4; News Satire / Fabrication: 4.

### Logistic Regression: per-class test results

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Native Advertising | 98.6842% | 100.0000% | 99.3377% | 225 |
| News Satire | 94.9541% | 92.0000% | 93.4537% | 225 |
| Propaganda | 100.0000% | 30.2222% | 46.4164% | 225 |
| Manipulation | 100.0000% | 99.5556% | 99.7773% | 225 |
| News Parody | 57.1809% | 95.5556% | 71.5474% | 225 |
| Fabrication | 83.2258% | 89.5833% | 86.2876% | 144 |

Test mistakes: **201/1269**. Matrix rows are true categories and columns predicted categories; IDs follow the final 1?6 order.

| True / predicted | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 225 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 207 | 0 | 0 | 12 | 6 |
| 3 | 1 | 1 | 68 | 0 | 143 | 12 |
| 4 | 0 | 0 | 0 | 224 | 1 | 0 |
| 5 | 1 | 1 | 0 | 0 | 215 | 8 |
| 6 | 1 | 9 | 0 | 0 | 5 | 129 |

Most confused category pairs (both directions combined): Propaganda / News Parody: 143; News Satire / Fabrication: 15; News Satire / News Parody: 13.


## Fixed 12-example qualitative check

| Example | Intended | BERT prediction (confidence) | DistilBERT prediction (confidence) | LR prediction (confidence) |
|---:|---|---|---|---|
| 1 | Native Advertising | 3: Propaganda (99.3375%; ERROR) | 3: Propaganda (99.7118%; ERROR) | 6: Fabrication (42.9164%; ERROR) |
| 2 | Native Advertising | 1: Native Advertising (96.9908%; match) | 1: Native Advertising (67.8668%; match) | 1: Native Advertising (34.7199%; match) |
| 3 | News Satire | 5: News Parody (99.5624%; ERROR) | 5: News Parody (93.5798%; ERROR) | 5: News Parody (57.8397%; ERROR) |
| 4 | News Satire | 2: News Satire (99.7063%; match) | 2: News Satire (99.6475%; match) | 2: News Satire (67.1069%; match) |
| 5 | Propaganda | 3: Propaganda (99.4583%; match) | 3: Propaganda (98.5150%; match) | 5: News Parody (23.1574%; ERROR) |
| 6 | Propaganda | 3: Propaganda (99.5797%; match) | 3: Propaganda (99.0972%; match) | 1: Native Advertising (33.9504%; ERROR) |
| 7 | Manipulation | 4: Manipulation (99.7445%; match) | 4: Manipulation (99.7246%; match) | 4: Manipulation (85.9873%; match) |
| 8 | Manipulation | 4: Manipulation (99.7138%; match) | 4: Manipulation (99.6824%; match) | 4: Manipulation (56.0404%; match) |
| 9 | News Parody | 5: News Parody (99.6429%; match) | 5: News Parody (99.5646%; match) | 5: News Parody (79.0933%; match) |
| 10 | News Parody | 5: News Parody (99.6241%; match) | 5: News Parody (99.5793%; match) | 5: News Parody (83.6127%; match) |
| 11 | Fabrication | 6: Fabrication (99.3836%; match) | 6: Fabrication (99.6447%; match) | 5: News Parody (36.3277%; ERROR) |
| 12 | Fabrication | 6: Fabrication (99.1533%; match) | 6: Fabrication (98.8580%; match) | 5: News Parody (76.4691%; ERROR) |

Intended-category matches: BERT: 10/12; DistilBERT: 10/12; Logistic Regression: 6/12.

The first Native Advertising example (sponsored running shoes) is still misclassified: BERT as Propaganda at 99.3375%, DistilBERT as Propaganda at 99.7118%, and LR as Fabrication at 42.9164%. TravelPro is correctly classified by all three. These are model generalization limitations; no rules, training changes or tuning were added to force these probes to pass. Full original texts and all probabilities are saved in `results/final_model_comparison/representative_predictions.json`.

## Verification

- Syntax checks passed for all 10 changed/new Python files (AST parsing without writing bytecode).
- **40 offline tests passed**, zero failures, errors or skips; 66.79 seconds. Evidence: `offline_tests.log` and `offline_tests.json`.
- Coverage: all three saved classifiers, exact label/probability mappings, best BERT checkpoint integrity, TravelPro direct/module/Streamlit parity, cached evaluation-mode models, 512/513-token boundary and complete BERT/DistilBERT token coverage, long inference, agreement outcomes, Analyze/Clear, edited-input invalidation, empty/error states, and independent VADER positive/neutral/negative scoring.
- All 46 protected dataset/corrected-DistilBERT artifacts still match their original SHA-256 hashes. No protected artifact changed.
- BERT canonical weights are byte-for-byte identical to checkpoint 742. The restored checkpoint reproduced the original recorded best validation metrics; final test metrics remained unchanged.
- The test/parity processes returned success but Python emitted a Windows permission error during temporary-directory cleanup at interpreter exit. This did not fail any assertion; it is recorded as an environment limitation rather than hidden.
- Live HTTP/WebSocket interaction evidence is recorded in `live_streamlit_checks.json`: local server health, all six labels, three TravelPro predictions, independent VADER, two windows for each Transformer on long input, and successful Clear.

