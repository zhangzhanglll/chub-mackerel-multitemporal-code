#!/usr/bin/env python3
"""Spearman lag-correlation analysis with p-values and BH-FDR correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ENV_FEATURES = ["sla", "chla", "sss", "sst", "ugos", "vgos", "DO", "MLD", "CV"]
TARGET_COL = "鲐鱼/t"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Spearman correlations between catch and lagged environmental features."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Local raw xlsx/csv data path. Data files are not included in this repository.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/tables/spearman_lag_analysis",
        help="Directory for Spearman lag-correlation outputs.",
    )
    parser.add_argument(
        "--grid_res",
        type=float,
        default=0.5,
        help="Spatial grid resolution used for past-only rolling windows.",
    )
    parser.add_argument(
        "--lags",
        default="0-30",
        help="Lag days, e.g. 0-30 or 0,3,7,14,30. Use 0 for instantaneous values.",
    )
    parser.add_argument(
        "--target_col",
        default=TARGET_COL,
        help="Continuous target column for Spearman correlation.",
    )
    parser.add_argument(
        "--mode",
        choices=["exact_lag", "rolling_window"],
        default="exact_lag",
        help=(
            "exact_lag pairs the target at day t with the environmental value at t-lag; "
            "rolling_window uses past-only rolling means over selected lag windows."
        ),
    )
    return parser.parse_args()


def parse_lags(spec: str) -> list[int]:
    parts = [x.strip() for x in spec.split(",") if x.strip()]
    lags = []
    for part in parts:
        if "-" in part:
            start, end = part.split("-", 1)
            lags.extend(range(int(start), int(end) + 1))
        else:
            lags.append(int(part))
    if any(lag < 0 for lag in lags):
        raise SystemExit("Lag windows must be non-negative integers.")
    return sorted(set(lags))


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def add_grid_and_date(df: pd.DataFrame, grid_res: float) -> pd.DataFrame:
    df = df.copy()
    if {"latitude", "longitude"}.issubset(df.columns):
        df["lon_bin"] = (np.floor(df["longitude"] / grid_res) * grid_res).round(6)
        df["lat_bin"] = (np.floor(df["latitude"] / grid_res) * grid_res).round(6)
    elif not {"lat_bin", "lon_bin"}.issubset(df.columns):
        raise SystemExit("Input must contain either latitude/longitude or lat_bin/lon_bin columns.")

    df["grid_id"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)

    if {"year", "month", "day"}.issubset(df.columns):
        df["full_date"] = pd.to_datetime(df[["year", "month", "day"]])
    elif "full_date" in df.columns:
        df["full_date"] = pd.to_datetime(df["full_date"])
    elif "date" in df.columns:
        df["full_date"] = pd.to_datetime(df["date"])
    else:
        raise SystemExit("Input must contain year/month/day, full_date, or date columns.")

    missing_features = [col for col in ENV_FEATURES if col not in df.columns]
    if missing_features:
        raise SystemExit(f"Missing environmental feature columns: {missing_features}")

    return df.sort_values(["grid_id", "full_date"])


def add_rolling_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    lag_windows = [lag for lag in lags if lag > 0]

    def add_lags(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("full_date").copy()
        for col in ENV_FEATURES:
            history = group[col].shift(1)
            for lag in lag_windows:
                group[f"{col}_{lag}d"] = history.rolling(window=lag, min_periods=lag).mean()
        return group

    if lag_windows:
        df = pd.concat(
            [add_lags(group) for _, group in df.groupby("grid_id", sort=False)],
            ignore_index=True,
        )
    return df


def build_exact_lag_pairs(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    if target_col not in df.columns:
        raise SystemExit(f"Missing target column: {target_col}")

    value_cols = ["grid_id", "full_date", target_col, *ENV_FEATURES]
    daily = df[value_cols].copy()
    for col in [target_col, *ENV_FEATURES]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily = daily.groupby(["grid_id", "full_date"], as_index=False).mean(numeric_only=True)

    target_df = daily[["grid_id", "full_date", target_col]].rename(
        columns={"full_date": "target_date"}
    )
    env_df = daily[["grid_id", "full_date", *ENV_FEATURES]].rename(
        columns={"full_date": "env_date"}
    )

    parts = []
    for lag in lags:
        current = target_df.copy()
        current["env_date"] = current["target_date"] - pd.to_timedelta(lag, unit="D")
        current = current.merge(env_df, on=["grid_id", "env_date"], how="inner")
        current["lag_days"] = lag
        parts.append(current)

    if not parts:
        return pd.DataFrame(columns=["grid_id", "target_date", "env_date", target_col, "lag_days"])
    return pd.concat(parts, ignore_index=True)


def bh_fdr(p_values: list[float]) -> list[float]:
    m = len(p_values)
    adjusted = [np.nan] * m
    valid = [i for i, p in enumerate(p_values) if not pd.isna(p)]
    if not valid:
        return adjusted

    ordered = sorted(valid, key=lambda i: p_values[i])
    running_min = 1.0
    for rank_from_end, idx in enumerate(reversed(ordered), start=1):
        rank = len(valid) - rank_from_end + 1
        value = min(1.0, p_values[idx] * len(valid) / rank)
        running_min = min(running_min, value)
        adjusted[idx] = running_min
    return adjusted


def spearman_ci95(rho: float, n: int) -> tuple[float, float]:
    """Approximate 95% CI for Spearman rho using Fisher's z transform."""
    if pd.isna(rho) or n <= 3:
        return np.nan, np.nan
    clipped = float(np.clip(rho, -0.999999, 0.999999))
    z = np.arctanh(clipped)
    half_width = 1.959963984540054 / np.sqrt(n - 3)
    return float(np.tanh(z - half_width)), float(np.tanh(z + half_width))


def spearman_lag_table(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    if target_col not in df.columns:
        raise SystemExit(f"Missing target column: {target_col}")

    rows = []
    for feature in ENV_FEATURES:
        for lag in lags:
            feature_col = feature if lag == 0 else f"{feature}_{lag}d"
            if feature_col not in df.columns:
                continue
            pair = df[[target_col, feature_col]].apply(pd.to_numeric, errors="coerce").dropna()
            n = len(pair)
            if n < 3 or pair[target_col].nunique() < 2 or pair[feature_col].nunique() < 2:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(pair[target_col], pair[feature_col])
            ci_low, ci_high = spearman_ci95(rho, n)
            rows.append(
                {
                    "variable": feature,
                    "lag_days": lag,
                    "feature_col": feature_col,
                    "target_col": target_col,
                    "n": n,
                    "spearman_rho": float(rho) if not pd.isna(rho) else np.nan,
                    "spearman_rho_ci95_low": ci_low,
                    "spearman_rho_ci95_high": ci_high,
                    "p_value": float(p_value) if not pd.isna(p_value) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out["p_fdr_bh"] = bh_fdr(out["p_value"].tolist())
    out["significant_fdr_0.05"] = out["p_fdr_bh"].lt(0.05)
    out["abs_spearman_rho"] = out["spearman_rho"].abs()
    out = out.sort_values(
        ["p_fdr_bh", "p_value", "abs_spearman_rho", "variable", "lag_days"],
        ascending=[True, True, False, True, True],
        na_position="last",
    )
    return out


def spearman_exact_lag_table(pairs: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    rows = []
    for feature in ENV_FEATURES:
        for lag in lags:
            pair = pairs[pairs["lag_days"] == lag][[target_col, feature]].dropna()
            n = len(pair)
            if n < 3 or pair[target_col].nunique() < 2 or pair[feature].nunique() < 2:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(pair[target_col], pair[feature])
            ci_low, ci_high = spearman_ci95(rho, n)
            rows.append(
                {
                    "variable": feature,
                    "lag_days": lag,
                    "feature_col": feature,
                    "target_col": target_col,
                    "n": n,
                    "spearman_rho": float(rho) if not pd.isna(rho) else np.nan,
                    "spearman_rho_ci95_low": ci_low,
                    "spearman_rho_ci95_high": ci_high,
                    "p_value": float(p_value) if not pd.isna(p_value) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out["p_fdr_bh"] = bh_fdr(out["p_value"].tolist())
    out["significant_fdr_0.05"] = out["p_fdr_bh"].lt(0.05)
    out["abs_spearman_rho"] = out["spearman_rho"].abs()
    out = out.sort_values(
        ["p_fdr_bh", "p_value", "abs_spearman_rho", "variable", "lag_days"],
        ascending=[True, True, False, True, True],
        na_position="last",
    )
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lags = parse_lags(args.lags)
    df = read_input(input_path)
    df = add_grid_and_date(df, args.grid_res)
    if args.mode == "exact_lag":
        pairs = build_exact_lag_pairs(df, args.target_col, lags)
        table = spearman_exact_lag_table(pairs, args.target_col, lags)
        stem = "spearman_exact_lag_correlations_fdr"
    else:
        df = add_rolling_lags(df, lags)
        table = spearman_lag_table(df, args.target_col, lags)
        stem = "spearman_rolling_window_correlations_fdr"

    table.to_csv(out_dir / f"{stem}.csv", index=False)
    table.sort_values(["variable", "lag_days"]).to_csv(
        out_dir / f"{stem}_by_variable.csv", index=False
    )

    print(f"Wrote {len(table)} Spearman lag tests to {out_dir}")
    print(table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
