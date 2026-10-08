# Business Entity Resolution Pipeline

This repository contains the end-to-end machine learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

## 1. Pipeline Overview
The solution resolves noisy, incomplete business entity fragments across multiple data sources into unified reference entities using a high-recall multi-key blocking engine combined with an $F_{0.5}$-optimized LightGBM classifier.

- **Data Partitioning**: Ground-truth cross-source validation confirmed that matching is strictly intra-country. The pipeline partitions data by country (`US`, `India`, `France`), optimizing memory footprint and indexing throughput.
- **Candidate Generation (Blocking)**:
  - Sorted token keys with legal suffix removal
  - Alphanumeric domain / web handle normalization
  - Numeric address anchors + street keyword pairing
  - Exact cleaned address matching
- **Feature Engineering**:
  - Token sort ratio, token set ratio, Levenshtein ratio, partial ratio (RapidFuzz)
  - Token Jaccard similarity & character n-gram overlap
  - Address number intersection ratio & missing address indicator
  - Composite similarity interaction terms (`max_sim`, `prod_sim`)
- **Matching Model & Decision Calibration**:
  - Gradient Boosted Decision Tree (LightGBM)
  - Decision threshold calibrated specifically for the precision-heavy **Macro $F_{0.5}$** metric.

## 2. Environment Setup

```bash
pip install -r requirements.txt
```

## 3. How to Run the End-to-End Pipeline

To reproduce the entire pipeline and generate both `output/matching_results.tsv` and `output/candidate_pairs.tsv`:

From the `student_resource/` directory:

```bash
python -m code.business_entity_resolution.src.main
```

## 4. Validation

To validate the generated output files against formatting, completeness, and consistency rules:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
