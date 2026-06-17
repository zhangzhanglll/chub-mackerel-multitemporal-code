# Input Schema

The repository does not include data. The scripts expect a local processed CSV with one row per fishing-log observation after environmental matching.

Recommended columns:

- `year`, `month`, `day`: date components.
- `full_date`: date string or datetime-compatible field.
- `lon_bin`, `lat_bin`: spatial grid coordinates used to build spatial groups.
- `grid_id`: optional spatial-grid identifier.
- `label`: binary target, where `1` indicates high-catch and `0` indicates low-catch.
- Instantaneous environmental variables, for example:
  - `sla`
  - `chla`
  - `sss`
  - `sst`
  - `ugos`
  - `vgos`
  - `DO`
  - `MLD`
  - `CV`
- Seasonal cyclic encodings:
  - `month_sin`
  - `month_cos`
- 7-day rolling means:
  - variables ending in `_7d`
- 30-day rolling means:
  - variables ending in `_30d`

The scripts infer feature sets from column suffixes:

- Base features without `_7d` or `_30d` are used for `E1`.
- Base features plus `_7d` features are used for `E3`.
- Base features plus `_30d` features are used for `E4`.
- Base features plus `_7d` and `_30d` features are used for `E2`.

Do not upload the processed CSV or any generated result file containing row counts, fold sizes, sample identifiers, or sample-level predictions.
