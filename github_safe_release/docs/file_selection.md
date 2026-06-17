# File Selection Notes

This safety-oriented release includes only code that follows the manuscript's main workflow:

1. Pre-defined `E1`/`E2`/`E3`/`E4` feature sets based on instantaneous, 7-day, and 30-day environmental variables.
2. Independent hyperparameter tuning for each model and feature set.
3. Nested spatial cross-validation.
4. Fold-level statistical comparison.
5. Global SHAP interpretation and Spearman lag diagnostic analysis.

The following file types are intentionally excluded:

- Raw data and processed sample-level data.
- Fold-level result CSV files.
- Sample-fold assignment files.
- Sample-level prediction files.
- Generated figures.
- Logs and temporary outputs.
- Exploratory sensitivity scripts involving alternative sample definitions, old feature windows, or seed/sample-count trials.

This avoids exposing sample counts, sample identifiers, or exploratory analyses that are not part of the manuscript's main experimental design.
