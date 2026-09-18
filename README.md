# Data-mining
# Data Mining Lab 1

This repository contains the implementation and outputs for Question 1 of the data mining lab, focused on data ingestion, duplicate handling, object-store partitioning, dimensional modeling, temporal price resolution, and federated analytics.

## Project Overview

The objective of this task is to process retail sales exports from multiple store systems, remove duplicate and resend rows, store clean data in a partitioned object store, and build a star schema for analysis. The workflow also includes historical price validation and a federated query over the warehouse model.

## Key Source Notes

Important operational context is documented in [Question_1/billing_notes.md](Question_1/billing_notes.md), including:

- daily export structure
- resend handling
- duplicate line detection logic
- product-code reissue issue
- temporal price rules
- file format differences across stores

## Question 1 Output Summary

The complete output record for this task is available in [Question_1/TASK_1 out put.txt](Question_1/TASK_1%20out%20put.txt).

### 1) Data Ingestion and Partitioned Object Store

- Raw rows: 1,137,585
- Rows removed as duplicate/resend lines: 16,661
- Final rows: 1,120,924
- Partition structure: store_id -> year -> month
- Output storage: partitioned Parquet files in the object store

### 2) Idempotent Ingestion and Duplicate Verification

- Total rows in final object store: 1,120,924
- Unique lines: 1,120,924
- Result: no duplicate lines remain after ingestion

### 3) Star Schema / Dimension Model

Dimension tables created:

- dim_date: 366 rows
- dim_store: 12 rows
- dim_category: 14 rows
- dim_product: 1,224 rows

Fact table:

- fact_sales

The model resolves product identity using:
- product_code + business date
- valid_from / valid_to validation

This ensures retired and reissued product codes are handled correctly.

### 4) Temporal Price Handling

Historical pricing is managed using the valid price revision for each reporting date.

Example reporting dates used in validation:

- 2024-03-15
- 2024-12-15

This ensures that the correct selling price is selected based on the effective time window rather than the printed price on the bill.

### 5) Federated Query

The final solution supports cross-dimension analytical queries joining:

- fact_sales
- dim_store
- dim_category
- dim_date

This allows revenue analysis by store, category, month, and day while respecting pricing and product validity rules.

## Repository Contents

- [Question_1/ingest.py](Question_1/ingest.py) — ingestion and partitioning
- [Question_1/check_duplicates.py](Question_1/check_duplicates.py) — duplicate validation
- [Question_1/create_star_schema.sql](Question_1/create_star_schema.sql) — schema definition
- [Question_1/masters.sql](Question_1/masters.sql) — master data setup
- [Question_1/populate_fact.py](Question_1/populate_fact.py) — fact population
- [Question_1/fast_fact.py](Question_1/fast_fact.py) — faster fact generation logic
- [Question_1/reconcile.py](Question_1/reconcile.py) — data reconciliation
- [Question_1/federated_query.py](Question_1/federated_query.py) — final analytical query
- [Question_1/visualization.py](Question_1/visualization.py) — visualization code

## Final Result

The final ingestion and warehouse pipeline successfully processes the full retail dataset, removes redundant resend lines, creates a valid dimensional model, resolves historical pricing correctly, and supports federated analytical reporting with consistent business logic.

---

This project was implemented as part of the Advanced Data Mining and Modern Warehousing lab.

===============================================================================================================================================


# Tender Duplicate Detection

This project focuses on detecting near-duplicate public procurement tender notices using text similarity, dimensionality reduction, candidate retrieval, and efficient indexing. The goal is to identify duplicate or highly similar notices across a large corpus while keeping computational cost manageable.

## Dataset Summary

- Total notices: 12,000
- Labelled pairs: 900
- Same pairs: 279
- Different pairs: 621
- Number of portals: 260
- Largest portal size: 1,426
- Smallest portal size: 2

## Methodology

The workflow follows a practical duplicate-detection pipeline:

1. Text preprocessing and TF-IDF vectorization
2. Similarity scoring using cosine distance
3. Dimensionality reduction using SVD
4. Candidate retrieval with nearest-neighbor filtering
5. SQLite-based storage and indexed queries
6. Full-corpus evaluation and skew analysis

## Key Results

### Part A: Baseline Similarity Model

Best threshold based on F1: 0.60

| Metric | Value |
|---|---:|
| Accuracy | 0.994444 |
| Precision | 0.989286 |
| Recall | 0.992832 |
| F1 Score | 0.991055 |

Confusion matrix:
- True negatives: 618
- False positives: 3
- False negatives: 2
- True positives: 277

### Part B: Reduced Representation

Best threshold based on F1: 0.65

| Metric | Value |
|---|---:|
| Accuracy | 0.967778 |
| Precision | 0.934028 |
| Recall | 0.964158 |
| F1 Score | 0.948854 |

Reduction:
- Original dimensions: 5000
- Reduced dimensions: 100
- Reduction: 98%

### Part C: Sublinear Retrieval

| Candidate size | Retrieval recall | Time (seconds) |
|---|---:|---:|
| 10 | 0.953405 | 28.858746 |
| 25 | 0.992832 | 29.819661 |
| 50 | 0.992832 | 28.097471 |
| 100 | 0.992832 | 29.184785 |
| 200 | 0.992832 | 29.011813 |
| 500 | 0.992832 | 28.051266 |

This shows that high recall can be preserved with a much smaller candidate set, reducing the need for exhaustive pairwise comparison.

### Part D: SQLite Storage and Indexing

| Metric | Value |
|---|---:|
| Rows stored | 12,000 |
| Database size | 61.38 MB |
| Average indexed lookup | 0.0496 ms |
| Index creation time | 0.0920 s |
| Database insertion time | 0.2047 s |

### Part E: Full-Corpus Experiment

- Runtime before mitigation: 4.6075 seconds
- Reduced representation shape: (12000, 100)
- Explained variance: 0.549169
- Portal skew: present across 260 portals, with a few portals dominating the corpus

## Main Files

- `solution.py` — baseline duplicate detection
- `part_b.py` — SVD-based reduced representation
- `part_c.py` — sublinear retrieval optimization
- `part_d.py` — SQLite index and storage layer
- `part_e.py` — full-corpus evaluation
- `summary_dashboard.py` — summary visualizations
- `Task2_output.txt` — full output log
- `portal_profiles.md` — data source and portal behavior notes

## Interpretation

The baseline TF-IDF cosine-similarity model gives the best overall performance, with an F1 score of 0.991. The SVD approach remains competitive while reducing dimensionality by 98%, making it useful for large-scale pipelines. Sublinear retrieval maintains near-perfect recall while reducing comparison cost, and SQLite indexing makes lookup operations highly efficient.

## Run Instructions

```bash
py solution.py
py part_b.py
py part_c.py
py part_d.py
py part_e.py
