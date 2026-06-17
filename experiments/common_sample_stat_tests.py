#!/usr/bin/env python3
"""Statistical summaries for common-sample nested spatial CV results.

Outputs:
- metrics_with_95CI.csv
- feature_set_wilcoxon_tests.csv
- model_wilcoxon_tests.csv
- significant_comparisons_summary.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon


METRICS = ["AUC", "F1", "Precision", "Recall", "PR_AUC"]
FEATURE_ORDER = ["E1", "E2", "E3", "E4"]
MODEL_ORDER = ["DF", "RF", "XGB", "LGBM", "CAT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="95% CI and paired Wilcoxon tests for common-sample nested CV."
    )
    parser.add_argument("--input_csv", required=True, help="Fold-level metrics CSV.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--default_model",
        default="DF",
        help="Model name to use when the input has no model column.",
    )
    parser.add_argument(
        "--apply_e1_fold1_update",
        action="store_true",
        help="Replace E1 fold 1 metrics with the user-specified updated values.",
    )
    return parser.parse_args()


def normalize_fold_metrics(df: pd.DataFrame, default_model: str) -> pd.DataFrame:
    df = df.copy()
    rename = {
        "E_set": "E_set",
        "feature_set": "E_set",
        "outer_fold": "outer_fold",
        "fold": "outer_fold",
        "test_auc": "AUC",
        "auc": "AUC",
        "AUC": "AUC",
        "test_f1": "F1",
        "f1": "F1",
        "F1": "F1",
        "test_precision": "Precision",
        "precision": "Precision",
        "Precision": "Precision",
        "test_recall": "Recall",
        "recall": "Recall",
        "Recall": "Recall",
        "test_pr_auc": "PR_AUC",
        "pr_auc": "PR_AUC",
        "PR_AUC": "PR_AUC",
        "PR-AUC": "PR_AUC",
    }
    df = df.rename(columns={c: rename[c] for c in df.columns if c in rename})
    if "model" not in df.columns:
        df["model"] = default_model
    df["model"] = df["model"].replace({"DeepForest": "DF", "CatBoost": "CAT"})
    required = {"model", "E_set", "outer_fold", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns after normalization: {sorted(missing)}")
    cols = ["model", "E_set", "outer_fold", *METRICS]
    out = df[cols].copy()
    out["outer_fold"] = out["outer_fold"].astype(int)
    for metric in METRICS:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out


def apply_e1_fold1_update(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = (df["model"] == "DF") & (df["E_set"] == "E1") & (df["outer_fold"] == 1)
    if mask.sum() != 1:
        raise SystemExit(
            "Expected exactly one DF/E1/outer_fold=1 row for --apply_e1_fold1_update, "
            f"found {int(mask.sum())}."
        )
    updates = {
        "AUC": 0.7196,
        "F1": 0.6211,
        "Precision": 0.5350,
        "Recall": 0.6192,
        "PR_AUC": 0.6875,
    }
    for col, val in updates.items():
        df.loc[mask, col] = val
    return df


def metrics_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, e_set), group in df.groupby(["model", "E_set"], sort=False):
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(float)
            n = len(values)
            mean = float(np.mean(values)) if n else np.nan
            sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
            se = sd / np.sqrt(n) if n > 1 else np.nan
            tcrit = float(t.ppf(0.975, n - 1)) if n > 1 else np.nan
            margin = tcrit * se if n > 1 else np.nan
            rows.append(
                {
                    "model": model,
                    "E_set": e_set,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "sd": sd,
                    "ci95_lower": mean - margin if n > 1 else np.nan,
                    "ci95_upper": mean + margin if n > 1 else np.nan,
                    "ci95_margin": margin,
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (np.inf if pd.isna(p_values[i]) else p_values[i]))
    adjusted = [np.nan] * m
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        p = p_values[idx]
        if pd.isna(p):
            adjusted[idx] = np.nan
            continue
        val = min(1.0, (m - rank + 1) * p)
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    diff = x - y
    if len(diff) == 0:
        return np.nan, np.nan
    if np.allclose(diff, 0):
        return 0.0, 1.0
    stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p)


def feature_set_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, model_df in df.groupby("model", sort=False):
        for metric in METRICS:
            metric_rows = []
            available = [e for e in FEATURE_ORDER if e in set(model_df["E_set"])]
            for a, b in combinations(available, 2):
                a_df = model_df[model_df["E_set"] == a][["outer_fold", metric]]
                b_df = model_df[model_df["E_set"] == b][["outer_fold", metric]]
                merged = a_df.merge(b_df, on="outer_fold", suffixes=("_a", "_b"))
                x = merged[f"{metric}_a"].to_numpy(float)
                y = merged[f"{metric}_b"].to_numpy(float)
                stat, p = paired_wilcoxon(x, y)
                metric_rows.append(
                    {
                        "comparison_type": "feature_set",
                        "model": model,
                        "metric": metric,
                        "group_a": a,
                        "group_b": b,
                        "n_pairs": len(merged),
                        "mean_a": float(np.mean(x)) if len(x) else np.nan,
                        "mean_b": float(np.mean(y)) if len(y) else np.nan,
                        "mean_diff_a_minus_b": float(np.mean(x - y)) if len(x) else np.nan,
                        "wilcoxon_stat": stat,
                        "p_value": p,
                    }
                )
            adj = holm_adjust([r["p_value"] for r in metric_rows])
            for r, p_adj in zip(metric_rows, adj):
                r["p_holm"] = p_adj
                r["significant_0.05"] = bool(p_adj < 0.05) if not pd.isna(p_adj) else False
            rows.extend(metric_rows)
    return pd.DataFrame(rows)


def model_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for e_set, e_df in df.groupby("E_set", sort=False):
        for metric in METRICS:
            metric_rows = []
            available = [m for m in MODEL_ORDER if m in set(e_df["model"])]
            for a, b in combinations(available, 2):
                a_df = e_df[e_df["model"] == a][["outer_fold", metric]]
                b_df = e_df[e_df["model"] == b][["outer_fold", metric]]
                merged = a_df.merge(b_df, on="outer_fold", suffixes=("_a", "_b"))
                x = merged[f"{metric}_a"].to_numpy(float)
                y = merged[f"{metric}_b"].to_numpy(float)
                stat, p = paired_wilcoxon(x, y)
                metric_rows.append(
                    {
                        "comparison_type": "model",
                        "E_set": e_set,
                        "metric": metric,
                        "group_a": a,
                        "group_b": b,
                        "n_pairs": len(merged),
                        "mean_a": float(np.mean(x)) if len(x) else np.nan,
                        "mean_b": float(np.mean(y)) if len(y) else np.nan,
                        "mean_diff_a_minus_b": float(np.mean(x - y)) if len(x) else np.nan,
                        "wilcoxon_stat": stat,
                        "p_value": p,
                    }
                )
            adj = holm_adjust([r["p_value"] for r in metric_rows])
            for r, p_adj in zip(metric_rows, adj):
                r["p_holm"] = p_adj
                r["significant_0.05"] = bool(p_adj < 0.05) if not pd.isna(p_adj) else False
            rows.extend(metric_rows)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_df = normalize_fold_metrics(pd.read_csv(args.input_csv), args.default_model)
    if args.apply_e1_fold1_update:
        fold_df = apply_e1_fold1_update(fold_df)

    fold_df.to_csv(out_dir / "fold_level_metrics_used_for_tests.csv", index=False)
    ci_df = metrics_with_ci(fold_df)
    feature_df = feature_set_tests(fold_df)
    model_df = model_tests(fold_df)

    ci_df.to_csv(out_dir / "metrics_with_95CI.csv", index=False)
    feature_df.to_csv(out_dir / "feature_set_wilcoxon_tests.csv", index=False)
    model_df.to_csv(out_dir / "model_wilcoxon_tests.csv", index=False)

    sig_parts = []
    if not feature_df.empty:
        sig_parts.append(feature_df[feature_df["significant_0.05"]].copy())
    if not model_df.empty:
        sig_parts.append(model_df[model_df["significant_0.05"]].copy())
    if sig_parts:
        sig_df = pd.concat(sig_parts, ignore_index=True)
    else:
        sig_df = pd.DataFrame(
            columns=[
                "comparison_type",
                "model",
                "E_set",
                "metric",
                "group_a",
                "group_b",
                "n_pairs",
                "mean_a",
                "mean_b",
                "mean_diff_a_minus_b",
                "wilcoxon_stat",
                "p_value",
                "p_holm",
                "significant_0.05",
            ]
        )
    sig_df.to_csv(out_dir / "significant_comparisons_summary.csv", index=False)

    print("Wrote:", out_dir)
    print("CI rows:", len(ci_df))
    print("Feature-set test rows:", len(feature_df))
    print("Model test rows:", len(model_df))
    print("Significant rows:", len(sig_df))


if __name__ == "__main__":
    main()
