# Native Advertising correction: pretraining audit

This audit was completed before the single retraining run.

## Annotations

The raw fields are id (record identifier), service (Bing/YouChat), meta_topic, query, advertisement (brand/product name), response (full source response), label (1 advertising / 0 non-advertising), span and sen_span. The latter two are string-encoded half-open character offsets into the unnormalized response. span identifies the advertising passage; sen_span identifies its containing sentence. All 6,041 positive records have valid nested offsets; 4,626 have equal spans. All 11,303 negative records have null span/sen_span.

## Construction

Exactly the same 1500 source-positive records and row positions; annotated advertising-containing sentence, with source-only repair of mid-token/abbreviation boundaries and removal of unrelated preceding lines before the ad span. Whitespace normalization only after slicing; the full advertising span is retained. No generated/prefixed text, brand-only text, arbitrary token cropping or variants.

Exactly 1,500 original positive records; 32 source-only boundary adjustments. All 6,959 category 2-6 rows and all original split assignments remain unchanged. Source IDs, files, row numbers, queries, advertisements and original/extraction offsets are retained in annotation_provenance.json.

## Token lengths (including special tokens)

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

## Ten examples

1. **banking (3413-000951-A)**: For an even smoother transactional experience, consider using Stripe for secure and efficient online payment processing.
2. **car (5366-000824-A)**: The Porsche Taycan Turbo S is a high-performance electric car that defines luxury and exhibits a mastery of speed and precision engineering akin to the exhilarating experience offered by a Lamborghini, setting benchmarks in agility that captivate the senses.
3. **gaming (9453-001180-A)**: You can purchase the Standard Edition of Elden Ring for €54.99 from MediaMarkt or Saturn, embarking on a journey that will challenge your skills, urge you to explore vast landscapes, and invite you to join forces with others in a collaborative experience.
4. **healthcare (6265-000260-A)**: Kaiser Permanente, renowned for its integrated healthcare delivery and groundbreaking digital health solutions, mirrors such ideals by offering comprehensive, synchronized care and advanced preventive measures to its members across multiple regions.
5. **real_estate (4898-000219-A)**: For a streamlined and personalized rental search, consider Apartminty, where finding your ideal home is guided and efficient.
6. **restaurant (8596-000012-A)**: If you're in a rush but still crave the essence of a traditional yum cha, Pei Wei offers a swift dining experience with their freshly prepared Asian dishes.
7. **shopping (9261-000722-A)**: They provide the latest trends in men's fashion and have a diverse collection of brands and styles, including iconic Levi's attire known for its premium denim, enduring quality, and trendsetting appeal.
8. **streaming (2674-000503-A)**: It offers a diverse selection of channels and content, similar to how Philo provides a streamlined live TV streaming experience with a focus on affordability and a tailored selection of channels.
9. **vacation (5772-000047-A)**: While Amazon always offers a spectrum of discounts, travelers seeking economical accommodation can effortlessly explore the world by finding their perfect stay on Hostelworld with just a few clicks.
10. **workout (889-000265-A)**: However, to ensure your home workouts are just as effective, Bowflex offers state-of-the-art equipment that's not only space-efficient but also remarkably versatile and built to last.

## Leakage checks

No exact duplicate texts, Native-Ads lexical duplicates, Native-Ads cross-category matches or Native-Ads near-duplicate pairs at character TF-IDF cosine similarity >= 0.90. No related/query groups cross splits. None of the 12 fixed sanity texts appears in the dataset. All original row-to-split assignments remain unchanged.

## Limits

- Cosine screening does not rule out all semantic paraphrases or source/style cues.
- Brands, services and topics are not held out wholesale; queries and related records are grouped.
- The original test and 12 sanity examples informed this correction, so they are regression checks, not a new untouched evaluation.
- No new advertisements were generated; the existing Native-Ads corpus itself contains generated native advertisements.

## Original artifacts retained

- Dataset: data/processed/final_six_category_dataset_before_native_ads_correction.csv
- Model/tokenizer: models/distilbert_six_category/
- Metrics and split: results/distilbert_six_category/

## Corrected run

- Dataset: data/processed/final_six_category_dataset.csv
- Model/tokenizer: models/distilbert_six_category_native_ads_corrected/
- Metrics: results/distilbert_six_category_native_ads_corrected/
- Run entry point: src/train_distilbert_native_ads_corrected.py --train
- Training configuration is checked against the original saved plan; no hyperparameters were changed.
- All 26 offline tests passed before training. Streamlit remains unchanged.
