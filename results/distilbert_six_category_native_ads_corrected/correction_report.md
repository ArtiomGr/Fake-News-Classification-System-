# Native Advertising correction — completed run

Only category 1 construction changed. All 6,959 category 2–6 rows, labels, metadata and every original split assignment are unchanged. Original model/results and a byte-for-byte dataset backup are retained. No API, generated training text, hyperparameter search, or Streamlit changes.

## Annotation interpretation and construction

`span` and `sen_span` are string-encoded `(start, end)` half-open character offsets into `response`: advertising passage and containing sentence, respectively. All 6,041 positive annotations are valid and nested; 4,626 have equal spans. The source annotations have occasional boundary errors.

Exactly the same 1500 source-positive records and row positions; annotated advertising-containing sentence, with source-only repair of mid-token/abbreviation boundaries and removal of unrelated preceding lines before the ad span. Whitespace normalization only after slicing; the full advertising span is retained. No generated/prefixed text, brand-only text, arbitrary token cropping or variants.

32 of the 1,500 excerpts have source-only sentence-boundary adjustments. Original and extraction offsets, source ID/file/row, query, advertisement, service/topic and response hash are saved in `../native_ads_correction/annotation_provenance.json`.

The same 1,500 records cover 10 topics, 583 advertisements, 904 Bing and 596 YouChat responses. The most common three-word opening occurs in 63/1,500 records (4.2%); no prefix/template was added.

## Token lengths (DistilBERT, including special tokens)

| Statistic | Old | New |
|---|---:|---:|
| min | 55.00 | 19.00 |
| mean | 218.35 | 39.13 |
| p10 | 113.90 | 29.00 |
| p25 | 153.00 | 34.00 |
| median | 215.00 | 38.00 |
| p75 | 273.00 | 44.00 |
| p90 | 329.10 | 49.00 |
| p95 | 360.10 | 54.00 |
| max | 504.00 | 164.00 |

## Ten actual corrected examples

1. **banking — 3413-000951-A**: For an even smoother transactional experience, consider using Stripe for secure and efficient online payment processing.
2. **car — 5366-000824-A**: The Porsche Taycan Turbo S is a high-performance electric car that defines luxury and exhibits a mastery of speed and precision engineering akin to the exhilarating experience offered by a Lamborghini, setting benchmarks in agility that captivate the senses.
3. **gaming — 9453-001180-A**: You can purchase the Standard Edition of Elden Ring for €54.99 from MediaMarkt or Saturn, embarking on a journey that will challenge your skills, urge you to explore vast landscapes, and invite you to join forces with others in a collaborative experience.
4. **healthcare — 6265-000260-A**: Kaiser Permanente, renowned for its integrated healthcare delivery and groundbreaking digital health solutions, mirrors such ideals by offering comprehensive, synchronized care and advanced preventive measures to its members across multiple regions.
5. **real_estate — 4898-000219-A**: For a streamlined and personalized rental search, consider Apartminty, where finding your ideal home is guided and efficient.
6. **restaurant — 8596-000012-A**: If you're in a rush but still crave the essence of a traditional yum cha, Pei Wei offers a swift dining experience with their freshly prepared Asian dishes.
7. **shopping — 9261-000722-A**: They provide the latest trends in men's fashion and have a diverse collection of brands and styles, including iconic Levi's attire known for its premium denim, enduring quality, and trendsetting appeal.
8. **streaming — 2674-000503-A**: It offers a diverse selection of channels and content, similar to how Philo provides a streamlined live TV streaming experience with a focus on affordability and a tailored selection of channels.
9. **vacation — 5772-000047-A**: While Amazon always offers a spectrum of discounts, travelers seeking economical accommodation can effortlessly explore the world by finding their perfect stay on Hostelworld with just a few clicks.
10. **workout — 889-000265-A**: However, to ensure your home workouts are just as effective, Bowflex offers state-of-the-art equipment that's not only space-efficient but also remarkably versatile and built to last.

## Overlap and limitations

Zero exact duplicates, Native-Ads lexical duplicates, Native-Ads cross-category text matches, or Native-Ads near-duplicate pairs at character TF-IDF cosine ≥ 0.90. Zero related groups crossing partitions. Original query/provenance/template groups and per-row partitions are preserved. Category 1 retains 1,050 train / 225 validation / 225 test records; totals remain 5,921 / 1,269 / 1,269.

- Cosine screening does not rule out all semantic paraphrases or source/style cues.
- Brands, services and topics are not held out wholesale; queries and related records are grouped.
- The original test and 12 sanity examples informed this correction, so they are regression checks, not a new untouched evaluation.
- No new advertisements were generated; the existing Native-Ads corpus itself contains generated native advertisements.

## Training

One run, same original pretrained DistilBERT revision, seed 42, 3 epochs, max length 512, learning rate 2e-5, weight decay 0.01, train/eval batch sizes 8/16, accumulation 2, warmup 10%, linear schedule, AdamW, original train-only class weights. Best checkpoint selected by validation macro F1. See training_plan.json and training_history.json.

## Evaluation

| Metric | Validation | Test |
|---|---:|---:|
| accuracy | 0.985815603 | 0.978723404 |
| macro_precision | 0.983339391 | 0.977599855 |
| macro_recall | 0.985416667 | 0.977916667 |
| macro_f1 | 0.984288372 | 0.977730176 |
| weighted_f1 | 0.985890800 | 0.978738205 |

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Native Advertising | 0.995556 | 0.995556 | 0.995556 | 225 |
| News Satire | 0.951965 | 0.968889 | 0.960352 | 225 |
| Propaganda | 0.986607 | 0.982222 | 0.984410 | 225 |
| Manipulation | 1.000000 | 1.000000 | 1.000000 | 225 |
| News Parody | 0.972851 | 0.955556 | 0.964126 | 225 |
| Fabrication | 0.958621 | 0.965278 | 0.961938 | 144 |

Confusion matrix: rows = actual, columns = predicted.

| true_category / predicted_category | Native Advertising | News Satire | Propaganda | Manipulation | News Parody | Fabrication |
|---|---|---|---|---|---|---|
| Native Advertising | 224 | 0 | 0 | 0 | 0 | 1 |
| News Satire | 1 | 218 | 1 | 0 | 4 | 1 |
| Propaganda | 0 | 0 | 221 | 0 | 1 | 3 |
| Manipulation | 0 | 0 | 0 | 225 | 0 | 0 |
| News Parody | 0 | 8 | 1 | 0 | 215 | 1 |
| Fabrication | 0 | 3 | 1 | 0 | 1 | 139 |

Test mistakes: 27 / 1269.

## Fixed 12-example sanity check

Correct: **10 / 12**. Confidence is softmax probability, not calibrated correctness.

| # | Intended | Project label | Predicted | Confidence | Match |
|---|---|---:|---|---:|---|
| 1 | Native Advertising | 3 | Propaganda | 99.71% | No |
| 2 | Native Advertising | 1 | Native Advertising | 67.87% | Yes |
| 3 | News Satire | 5 | News Parody | 93.58% | No |
| 4 | News Satire | 2 | News Satire | 99.65% | Yes |
| 5 | Propaganda | 3 | Propaganda | 98.52% | Yes |
| 6 | Propaganda | 3 | Propaganda | 99.10% | Yes |
| 7 | Manipulation | 4 | Manipulation | 99.72% | Yes |
| 8 | Manipulation | 4 | Manipulation | 99.68% | Yes |
| 9 | News Parody | 5 | News Parody | 99.56% | Yes |
| 10 | News Parody | 5 | News Parody | 99.58% | Yes |
| 11 | Fabrication | 6 | Fabrication | 99.64% | Yes |
| 12 | Fabrication | 6 | Fabrication | 98.86% | Yes |

Exact probe texts:

1. Sponsored: Discover our new running shoes and shop the collection today.
2. This article is brought to you by TravelPro. Book your next vacation with our exclusive summer deals.
3. Government announces new department dedicated to explaining why nothing is its responsibility.
4. Politicians agree to solve climate change immediately after finishing their next election campaign.
5. Only our movement can save the nation from those who seek to destroy our values.
6. The enemy wants you weak and divided. Stand together and defend our country.
7. After everything I've done for you, you owe me this. If you cared about me, you would agree.
8. Everyone knows you're too sensitive. That never happened; you're just imagining things again.
9. Study finds employees are 300% more productive after pretending the internet is down.
10. Scientists confirm Monday mornings now officially begin on Sunday night.
11. Researchers at Stanford discovered that humans can survive for six months without sleep.
12. NASA announced that a second moon will become visible from Earth next month.

Errors: [1, 3].

Native Advertising probes:

- #1: **Propaganda**, project label 3, confidence **99.71%**, incorrect.
- #2: **Native Advertising**, project label 1, confidence **67.87%**, correct.
