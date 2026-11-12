# Cancer Biomarker Feature Selection

Ranking candidate biomarker genes in public microarray gene-expression datasets, using cross-validated sparse and tree-based classifiers and **selection stability** across folds.

Senior project (CMPE 492), Boğaziçi University, Spring 2023, advised by Prof. Arzucan Özgür. The code was rewritten in 2026; the original notebook and script are kept in [`notebooks/`](notebooks/).

## Data

Not included in this repository. Use any binary dataset from the [Curated Microarray Database (CuMiDa)](https://sbcb.inf.ufrgs.br/cumida). Each CSV has one row per sample with the columns `samples`, `type` and one column per probe. The project used breast-cancer (e.g. GSE26910) and liver-cancer (GSE14520, GSE62232) sets. To report gene symbols instead of probe IDs, pass the matching GEO platform annotation table (e.g. GPL570), downloaded from GEO.

## Method

For each dataset and each classifier (L1-penalised logistic regression, random forest):

1. Repeated stratified *k*-fold cross-validation (default 3 folds × 10 repeats).
2. Inside every training fold: keep the 2,000 most variable probes, standardise, fit. Held-out samples never influence filtering or scaling.
3. Score the held-out fold (accuracy, balanced accuracy, ROC-AUC).
4. Record the fold's top-*k* probes by |coefficient| or impurity importance.
5. Rank probes by **selection frequency**, the fraction of folds in which they made the top-*k*.

With several datasets, the probes selected in at least half the folds of every dataset are reported.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py data/Breast_GSE26910.csv --out results/
python main.py data/Liver_GSE14520_U133A.csv data/Liver_GSE62232.csv --probe-map data/GPL570.txt
pytest -q          # synthetic-data tests
```

Outputs per dataset and classifier: `*_folds.csv` (fold metrics) and `*_ranking.csv` (probe ranking).

## Example result: GSE26910 (breast, 6 tumour vs 6 normal, 54,675 probes)

Run with the default settings (3 folds × 10 repeats, seed 0) on 2026-11-12:

| Classifier | Balanced accuracy | ROC-AUC | Highest selection frequency (top-20) |
|---|---|---|---|
| L1 logistic regression | 0.85 ± 0.18 | 0.96 ± 0.12 | 0.33 (`213900_at`) |
| Random forest | 0.94 ± 0.11 | 1.00 ± 0.00 | 0.13 (several probes) |

The two classes are easy to separate, but **no probe is selected consistently**: with 12 samples and over 54,000 probes, many different gene sets separate the classes equally well. Treat these rankings as hypotheses, not biomarkers; validating them needs larger cohorts or several datasets of the same cancer type.

## What changed from the 2023 version

The original analysis applied PCA before LIME, so its "important features" were principal-component thresholds (e.g. `PC0 <= -41.67`) rather than genes. It also fitted PCA on all samples before the train/test split, and its probe-to-gene mapping step overwrote expression values. The rewrite works on probes directly, keeps every fitted step inside the training folds, and replaces single splits with repeated cross-validation.
