# Multi-Scale Fishing Ground Prediction Experiments

This repository contains the code used for the manuscript experiments on multi-scale environmental features for high-catch fishing-ground prediction.

The raw fishery logbook data, processed sample-level data, fold-level results, figures, and any files containing sample counts or sample identifiers are not included because of data-sharing restrictions.

## Feature Sets

The workflow compares four pre-defined temporal feature sets:

- `E1`: instantaneous environmental variables plus seasonal cyclic encodings.
- `E3`: `E1` plus 7-day rolling environmental means.
- `E4`: `E1` plus 30-day rolling environmental means.
- `E2`: `E1` plus both 7-day and 30-day rolling environmental means.

Rolling features should be calculated using past observations only. The current day and future observations should not be included.

## Validation Design

The main experiments use nested spatial cross-validation:

- Outer split: spatial `GroupKFold`.
- Inner split: spatial `GroupKFold` on the outer-training subset.
- Hyperparameter selection: mean inner-validation AUC.
- Final evaluation: the outer-test fold only.

Each model and each feature set are tuned independently.

## Scripts

Main model evaluation:

- `experiments/deepforest_e1e4_nested_spatial.py`: Deep Forest nested spatial CV for `E1`-`E4`.
- `experiments/baseline_e1e4_nested_spatial.py`: RF, XGBoost, LightGBM, and CatBoost nested spatial CV for `E1`-`E4`.

Statistics and plotting:

- `experiments/common_sample_stat_tests.py`: 95% confidence intervals, paired Wilcoxon signed-rank tests, and Holm correction.
- `experiments/plot_outer_fold_distributions.py`: outer-fold AUC/F1 distribution plots.

Interpretation:

- `experiments/deepforest_representative_shap.py`: representative outer-fold SHAP global plots.
- `experiments/spearman_lag_analysis.py`: Spearman lag-correlation diagnostic analysis.

## Data

No data files are included. To run the scripts, prepare a local processed CSV with the columns described in `docs/input_schema.md`.

Do not commit generated files from `data/`, `results/`, `figures/`, or any sample-level output.

## Example Commands

Deep Forest:

```bash
python experiments/deepforest_e1e4_nested_spatial.py \
  --input_csv data/processed/master_0.5_7_30_allrows.csv \
  --out_dir results/tables/deepforest_e1e4 \
  --lon_col lon_bin \
  --lat_col lat_bin \
  --date_col full_date \
  --target_col label \
  --n_blocks 6 \
  --n_candidates 10 \
  --light_profile
```

Baseline models:

```bash
python experiments/baseline_e1e4_nested_spatial.py \
  --input_csv data/processed/master_0.5_7_30_allrows.csv \
  --out_dir results/tables/baseline_e1e4 \
  --lon_col lon_bin \
  --lat_col lat_bin \
  --date_col full_date \
  --target_col label \
  --n_blocks 6 \
  --n_candidates 10 \
  --models RF,XGB,LGBM,CAT \
  --light_profile
```

Statistical comparison:

```bash
python experiments/common_sample_stat_tests.py \
  --input_csv results/tables/fold_level_metrics.csv \
  --out_dir results/tables/stat_tests
```
