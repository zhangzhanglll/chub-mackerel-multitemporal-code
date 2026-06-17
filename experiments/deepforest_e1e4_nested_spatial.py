#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepForest E1-E4 under Nested Spatial CV (with optional random E1 baseline).

Input CSV should contain:
- coordinates: lon, lat
- time column: date (optional for this script's splitting logic)
- target: y_binary
- feature columns for E1-E4 patterns:
  base features + optional *_7d + optional *_30d

Outputs:
- metrics_table.csv
- (optional) compare_random_vs_spatial_E1.csv
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")

ROOT = Path(__file__).resolve().parents[1]
VENDOR_SITE = ROOT / "vendor" / "py310_site"
if VENDOR_SITE.exists() and str(VENDOR_SITE) not in sys.path:
    sys.path.insert(0, str(VENDOR_SITE))

# numpy>=1.24 compatibility
if not hasattr(np, "bool"):
    np.bool = bool

import deepforest._layer as df_layer
import deepforest.cascade as df_cascade
from deepforest import CascadeForestClassifier

DEFAULT_NON_FEATURE_COLS = [
    "year",
    "month",
    "day",
    "full_date",
    "grid_id",
    "sample_id",
    "id",
    "idx",
    "spatial_block",
]


def df_is_classifier(estimator):
    return getattr(estimator, "_estimator_type", None) == "classifier"


def apply_deepforest_compat_patch():
    df_layer.is_classifier = df_is_classifier
    df_cascade.is_classifier = df_is_classifier


apply_deepforest_compat_patch()


def parse_args():
    parser = argparse.ArgumentParser(description="DeepForest nested spatial CV for E1-E4")
    parser.add_argument("--input_csv", required=True, help="Input CSV path.")
    parser.add_argument("--out_dir", default="results/tables", help="Output directory.")
    parser.add_argument("--lon_col", default="lon", help="Longitude column name.")
    parser.add_argument("--lat_col", default="lat", help="Latitude column name.")
    parser.add_argument("--date_col", default="date", help="Date column name.")
    parser.add_argument("--target_col", default="y_binary", help="Binary target column.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n_blocks", type=int, default=10, help="KMeans blocks (8-15).")
    parser.add_argument("--n_candidates", type=int, default=120, help="Random param candidates.")
    parser.add_argument(
        "--feature_sets",
        default="E1,E2,E3,E4",
        help="Comma-separated feature sets to run, e.g. E2,E3,E4.",
    )
    parser.add_argument(
        "--e1_default_params_conservative",
        action="store_true",
        help=(
            "Use a single fixed conservative DeepForest parameter set for E1 "
            "instead of hyperparameter tuning."
        ),
    )
    parser.add_argument(
        "--fast_profile",
        action="store_true",
        help="Use a small DeepForest grid for full E1-E4 reruns on large data.",
    )
    parser.add_argument(
        "--light_profile",
        action="store_true",
        help="Use a shallow n10-friendly DeepForest grid after preliminary plateau checks.",
    )
    parser.add_argument("--patience_layers", type=int, default=3, help="Layer early-stop patience.")
    parser.add_argument("--delta_auc", type=float, default=0.001, help="Layer early-stop delta.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for F1/P/R.")
    parser.add_argument(
        "--exclude_cols",
        default="",
        help="Extra non-feature columns to exclude, comma-separated.",
    )
    parser.add_argument(
        "--check_time_leakage",
        action="store_true",
        help="Run basic time leakage sanity checks for *_7d/*_30d features.",
    )
    parser.add_argument(
        "--time_leak_sample_n",
        type=int,
        default=20,
        help="Sample rows to print for time leakage sanity checks.",
    )
    parser.add_argument(
        "--run_random_e1",
        action="store_true",
        help="Also run nested random CV baseline for E1.",
    )
    return parser.parse_args()


def parse_feature_sets_arg(value: str) -> list[str]:
    valid = {"E1", "E2", "E3", "E4"}
    feature_sets = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not feature_sets:
        raise SystemExit("--feature_sets produced an empty list.")
    bad = sorted(set(feature_sets) - valid)
    if bad:
        raise SystemExit(f"Unknown feature set(s): {bad}. Valid values are E1,E2,E3,E4.")
    return feature_sets


def e1_default_params_conservative() -> dict:
    return {
        "n_estimators": 2,
        "n_trees": 50,
        "max_layers": 6,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
    }


def build_param_space(fast_profile: bool = False, light_profile: bool = False):
    if fast_profile:
        return {
            "n_estimators": [2],
            "n_trees": [50],
            "max_layers": [4],
            "max_depth": [8],
            "min_samples_leaf": [3, 5],
            "max_features": ["sqrt"],
        }

    if light_profile:
        return {
            "n_estimators": [2, 3],
            "n_trees": [50, 100],
            "max_layers": [2, 4],
            "max_depth": [8, 10],
            "min_samples_leaf": [3, 5],
            "max_features": ["sqrt"],
        }

    # Medium-runtime profile: keep search diversity but cap expensive configs.
    return {
        "n_estimators": [2, 3, 4],
        "n_trees": [100, 200, 300],
        "max_layers": [8, 12],
        "max_depth": [None, 10],
        "min_samples_leaf": [3, 5],
        "max_features": ["sqrt", 0.5],
    }


def sample_param_candidates(grid: dict, n_candidates: int, seed: int):
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    if n_candidates >= len(combos):
        return combos
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(combos), size=n_candidates, replace=False)
    return [combos[i] for i in idx]


def infer_feature_sets(df: pd.DataFrame, drop_cols: set[str]):
    all_features = [c for c in df.columns if c not in drop_cols]
    feat_7d = sorted([c for c in all_features if c.endswith("_7d")])
    feat_30d = sorted([c for c in all_features if c.endswith("_30d")])
    base = sorted([c for c in all_features if c not in set(feat_7d + feat_30d)])

    return {
        "E1": base,
        "E2": base + feat_7d + feat_30d,
        "E3": base + feat_7d,
        "E4": base + feat_30d,
    }


def build_spatial_blocks(
    df: pd.DataFrame, lon_col: str, lat_col: str, n_blocks: int, seed: int = 42
):
    coords = df[[lon_col, lat_col]].values
    # Use numeric n_init for sklearn compatibility across 1.1+.
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10)
    return km.fit_predict(coords)


def make_model(params: dict, seed: int):
    apply_deepforest_compat_patch()
    allowed = set(inspect.signature(CascadeForestClassifier.__init__).parameters.keys())
    kwargs = dict(
        n_estimators=params["n_estimators"],
        n_trees=params["n_trees"],
        max_layers=params["max_layers"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        n_tolerant_rounds=params["max_layers"],
        delta=0.0,
        random_state=seed,
        n_jobs=-1,
        backend="sklearn",
        verbose=0,
    )
    # Some deepforest builds do not expose max_features in constructor.
    if "max_features" in allowed:
        kwargs["max_features"] = params["max_features"]
    return CascadeForestClassifier(**kwargs)


def predict_proba_at_layer(model, X, n_layers: int):
    old = int(model.n_layers_)
    model.n_layers_ = int(n_layers)
    try:
        return model.predict_proba(X)[:, 1]
    finally:
        model.n_layers_ = old


def safe_auc(y_true, prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, prob))


def select_layer_early_stop(model, X_val, y_val, patience: int, delta: float):
    auc_hist = []
    best_auc = -np.inf
    best_layer = 1
    built_layers = int(model.n_layers_)

    for layer in range(1, built_layers + 1):
        prob = predict_proba_at_layer(model, X_val, layer)
        auc = safe_auc(y_val, prob)
        if np.isnan(auc):
            continue
        auc_hist.append(auc)
        if auc > best_auc:
            best_auc = auc
            best_layer = layer

        # If AUC(layer_k) - AUC(layer_k-3) < 0.001, stop.
        if layer > patience and len(auc_hist) > patience:
            if auc_hist[-1] - auc_hist[-(patience + 1)] < delta:
                break

    if best_auc == -np.inf:
        return 1, 0.5
    return best_layer, best_auc


def metrics_from_prob(y_true, prob, threshold: float):
    pred = (prob >= threshold).astype(int)
    return {
        "auc": float(safe_auc(y_true, prob)) if len(np.unique(y_true)) >= 2 else 0.5,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)),
    }


def parse_extra_excludes(exclude_cols_arg: str) -> set[str]:
    if not exclude_cols_arg.strip():
        return set()
    return {c.strip() for c in exclude_cols_arg.split(",") if c.strip()}


def print_feature_set_info(feature_sets: dict[str, list[str]]):
    for e_set in ["E1", "E2", "E3", "E4"]:
        feats = feature_sets[e_set]
        print(f"{e_set}: n_features={len(feats)}, first_20={feats[:20]}")


def run_time_leakage_sanity_check(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str,
    target_col: str,
    lon_col: str,
    lat_col: str,
    sample_n: int,
    seed: int,
):
    print("rolling/lag 特征必须由 shift(1)+rolling 生成；本脚本不生成时序特征。")
    rolling_cols = [c for c in feature_cols if c.endswith("_7d") or c.endswith("_30d")]
    if not rolling_cols:
        print("Time leakage check: no *_7d/*_30d feature columns found, skip sampling.")
        return

    required_cols = [date_col, target_col, lon_col, lat_col] + rolling_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Time leakage check: missing columns {missing}, skip.")
        return

    probe = df[required_cols].dropna()
    if probe.empty:
        print("Time leakage check: no non-NA rows available, skip.")
        return

    n = min(sample_n, len(probe))
    sampled = probe.sample(n=n, random_state=seed)
    preview_cols = rolling_cols[:8]
    for _, row in sampled.iterrows():
        fvals = {c: row[c] for c in preview_cols}
        print(
            "[time_leak_sample]",
            {
                "date": str(row[date_col]),
                "y": int(row[target_col]),
                "lon": float(row[lon_col]),
                "lat": float(row[lat_col]),
                "feature_values_summary": fvals,
            },
        )

    if date_col in df.columns:
        date_series = pd.to_datetime(df[date_col], errors="coerce")
        for col in ["source_window_end", "window_end", "raw_date"]:
            if col in df.columns:
                wnd = pd.to_datetime(df[col], errors="coerce")
                valid = date_series.notna() & wnd.notna()
                if valid.any():
                    assert bool((wnd[valid] < date_series[valid]).all()), (
                        f"Time leakage check failed: require {col} < {date_col} on valid rows."
                    )
                    print(f"Time leakage assert passed: {col} < {date_col} on valid rows.")
                else:
                    print(f"Time leakage check: {col} has no valid comparable rows, skip assert.")


def nested_spatial_cv_eval(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_meta: pd.DataFrame,
    params_list: list[dict],
    n_blocks: int,
    seed: int,
    patience_layers: int,
    delta_auc: float,
    threshold: float,
    e_set_name: str,
):
    # outer: GroupKFold(K)
    outer_cv = GroupKFold(n_splits=n_blocks)
    rows = []
    tuning_rows = []
    sample_fold_rows = []

    for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y, groups=groups), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]
        g_tr = groups[tr_idx]
        g_te = groups[te_idx]
        meta_te = sample_meta.iloc[te_idx].copy()

        train_blocks = set(np.unique(g_tr).tolist())
        test_blocks = set(np.unique(g_te).tolist())
        intersection = train_blocks & test_blocks
        print(f"[outer_fold={outer_fold}] len(intersection)=={len(intersection)}")
        assert len(intersection) == 0, (
            f"Spatial leakage detected at outer fold {outer_fold}: "
            f"intersection={sorted(intersection)}"
        )

        best_param = None
        best_inner_auc = -np.inf
        best_inner_layer = 1

        # inner: GroupKFold(K-1) over outer-train groups
        n_unique = len(np.unique(g_tr))
        inner_splits = min(n_blocks - 1, n_unique)
        if inner_splits < 2:
            raise ValueError(
                f"outer_fold={outer_fold}: inner_splits={inner_splits} < 2. "
                f"Need at least 2 unique train blocks, got {n_unique}."
            )
        inner_cv = GroupKFold(n_splits=inner_splits)
        for param_idx, params in enumerate(params_list, start=1):
            aucs = []
            layers = []
            for in_tr_idx, in_va_idx in inner_cv.split(X_tr, y_tr, groups=g_tr):
                X_in_tr, y_in_tr = X_tr[in_tr_idx], y_tr[in_tr_idx]
                X_in_va, y_in_va = X_tr[in_va_idx], y_tr[in_va_idx]
                model = make_model(params, seed=seed)
                model.fit(X_in_tr, y_in_tr)
                layer, auc = select_layer_early_stop(
                    model, X_in_va, y_in_va, patience=patience_layers, delta=delta_auc
                )
                aucs.append(auc)
                layers.append(layer)

            mean_auc = float(np.mean(aucs)) if aucs else float("nan")
            mean_layer = int(round(float(np.mean(layers)))) if layers else 1
            print(
                f"[{e_set_name}][outer_fold={outer_fold}] "
                f"candidate {param_idx}/{len(params_list)} "
                f"mean_inner_auc={mean_auc:.4f}, mean_layer={mean_layer}, params={params}"
                ,
                flush=True,
            )
            tuning_rows.append(
                {
                    "E_set": e_set_name,
                    "outer_fold": outer_fold,
                    "candidate_idx": param_idx,
                    "n_candidates": len(params_list),
                    "mean_inner_auc": mean_auc,
                    "mean_layer": mean_layer,
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                }
            )
            if mean_auc > best_inner_auc:
                best_inner_auc = mean_auc
                best_param = params
                best_inner_layer = mean_layer
        print(
            f"[{e_set_name}][outer_fold={outer_fold}] selected "
            f"mean_inner_auc={best_inner_auc:.4f}, best_layer={best_inner_layer}, "
            f"best_param={best_param}"
            ,
            flush=True,
        )

        # retrain on full outer-train
        model = make_model(best_param, seed=seed)
        model.fit(X_tr, y_tr)
        selected_layer = min(best_inner_layer, int(model.n_layers_))
        prob_te = predict_proba_at_layer(model, X_te, selected_layer)
        m = metrics_from_prob(y_te, prob_te, threshold=threshold)
        rows.append(
            {
                "outer_fold": outer_fold,
                "fold_id": outer_fold,
                "train_size": int(len(tr_idx)),
                "test_size": int(len(te_idx)),
                "train_pos_rate": float(np.mean(y_tr)),
                "test_pos_rate": float(np.mean(y_te)),
                "n_train_blocks": int(len(train_blocks)),
                "n_test_blocks": int(len(test_blocks)),
                "train_blocks": json.dumps(sorted(train_blocks)),
                "test_blocks": json.dumps(sorted(test_blocks)),
                "intersection_len": int(len(intersection)),
                "best_param": json.dumps(best_param, ensure_ascii=False, sort_keys=True),
                "selected_layer": selected_layer,
                "test_auc": m["auc"],
                "test_f1": m["f1"],
                "test_precision": m["precision"],
                "test_recall": m["recall"],
                "test_pr_auc": m["pr_auc"],
            }
        )
        for _, meta_row in meta_te.iterrows():
            sample_fold_rows.append(
                {
                    "E_set": e_set_name,
                    "sample_id": meta_row["sample_id"],
                    "grid_id": meta_row.get("grid_id", ""),
                    "spatial_block": int(meta_row["spatial_block"]),
                    "fold_id": outer_fold,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(tuning_rows), pd.DataFrame(sample_fold_rows)


def nested_random_e1_eval(
    X: np.ndarray,
    y: np.ndarray,
    params_list: list[dict],
    n_splits: int,
    seed: int,
    patience_layers: int,
    delta_auc: float,
    threshold: float,
):
    rows = []
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        best_param = None
        best_inner_auc = -np.inf
        best_inner_layer = 1
        inner_cv = StratifiedKFold(n_splits=n_splits - 1, shuffle=True, random_state=seed + outer_fold)

        for params in params_list:
            aucs = []
            layers = []
            for in_tr_idx, in_va_idx in inner_cv.split(X_tr, y_tr):
                X_in_tr, y_in_tr = X_tr[in_tr_idx], y_tr[in_tr_idx]
                X_in_va, y_in_va = X_tr[in_va_idx], y_tr[in_va_idx]
                model = make_model(params, seed=seed)
                model.fit(X_in_tr, y_in_tr)
                layer, auc = select_layer_early_stop(
                    model, X_in_va, y_in_va, patience=patience_layers, delta=delta_auc
                )
                aucs.append(auc)
                layers.append(layer)

            mean_auc = float(np.mean(aucs))
            mean_layer = int(round(float(np.mean(layers))))
            if mean_auc > best_inner_auc:
                best_inner_auc = mean_auc
                best_param = params
                best_inner_layer = mean_layer

        model = make_model(best_param, seed=seed)
        model.fit(X_tr, y_tr)
        selected_layer = min(best_inner_layer, int(model.n_layers_))
        prob_te = predict_proba_at_layer(model, X_te, selected_layer)
        m = metrics_from_prob(y_te, prob_te, threshold=threshold)
        rows.append(
            {
                "outer_fold": outer_fold,
                "best_param": json.dumps(best_param, ensure_ascii=False, sort_keys=True),
                "selected_layer": selected_layer,
                "test_auc": m["auc"],
                "test_f1": m["f1"],
                "test_precision": m["precision"],
                "test_recall": m["recall"],
                "test_pr_auc": m["pr_auc"],
            }
        )
    return pd.DataFrame(rows)


def summarize_rows(rows: pd.DataFrame, e_set: str):
    return {
        "E_set": e_set,
        "AUC_mean": float(rows["test_auc"].mean()),
        "AUC_std": float(rows["test_auc"].std(ddof=1)),
        "F1_mean": float(rows["test_f1"].mean()),
        "F1_std": float(rows["test_f1"].std(ddof=1)),
        "Precision_mean": float(rows["test_precision"].mean()),
        "Precision_std": float(rows["test_precision"].std(ddof=1)),
        "Recall_mean": float(rows["test_recall"].mean()),
        "Recall_std": float(rows["test_recall"].std(ddof=1)),
        "PR_AUC_mean": float(rows["test_pr_auc"].mean()),
        "PR_AUC_std": float(rows["test_pr_auc"].std(ddof=1)),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    required = {args.lon_col, args.lat_col, args.target_col}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    # Keep a shared spatial block assignment across E1-E4.
    df = df.copy()
    if "sample_id" not in df.columns:
        df["sample_id"] = np.arange(len(df), dtype=int)
    df["spatial_block"] = build_spatial_blocks(
        df=df,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        n_blocks=args.n_blocks,
        seed=args.seed,
    )

    extra_excludes = parse_extra_excludes(args.exclude_cols)
    drop_cols = {
        args.lon_col,
        args.lat_col,
        args.date_col,
        args.target_col,
        *DEFAULT_NON_FEATURE_COLS,
        *extra_excludes,
    }
    feature_sets = infer_feature_sets(df, drop_cols=drop_cols)
    print_feature_set_info(feature_sets)

    if args.check_time_leakage:
        all_feats = sorted(set(feature_sets["E2"] + feature_sets["E3"] + feature_sets["E4"]))
        run_time_leakage_sanity_check(
            df=df,
            feature_cols=all_feats,
            date_col=args.date_col,
            target_col=args.target_col,
            lon_col=args.lon_col,
            lat_col=args.lat_col,
            sample_n=args.time_leak_sample_n,
            seed=args.seed,
        )

    params_list = sample_param_candidates(
        build_param_space(
            fast_profile=args.fast_profile,
            light_profile=args.light_profile,
        ),
        n_candidates=args.n_candidates,
        seed=args.seed,
    )

    metric_rows = []
    fold_detail_rows = []
    tuning_trace_rows = []
    sample_fold_rows = []
    requested_feature_sets = parse_feature_sets_arg(args.feature_sets)

    for e_set in requested_feature_sets:
        feats = feature_sets[e_set]
        if not feats:
            raise SystemExit(f"No features inferred for {e_set}. Check input columns.")
        if e_set == "E1" and args.e1_default_params_conservative:
            e_params_list = [e1_default_params_conservative()]
            print(
                f"{e_set}: using fixed conservative params instead of tuning: "
                f"{e_params_list[0]}",
                flush=True,
            )
        else:
            e_params_list = params_list
        work = df.dropna(subset=feats + [args.target_col]).copy()
        X = work[feats].values
        y = work[args.target_col].values
        g = work["spatial_block"].values
        sample_meta = work[["sample_id", "grid_id", "spatial_block"]].copy()
        if len(np.unique(g)) < args.n_blocks:
            raise SystemExit(
                f"{e_set}: unique spatial blocks in non-NA subset < n_blocks. "
                f"have={len(np.unique(g))}, need={args.n_blocks}"
            )

        rows, tuning_rows, sample_rows = nested_spatial_cv_eval(
            X=X,
            y=y,
            groups=g,
            sample_meta=sample_meta,
            params_list=e_params_list,
            n_blocks=args.n_blocks,
            seed=args.seed,
            patience_layers=args.patience_layers,
            delta_auc=args.delta_auc,
            threshold=args.threshold,
            e_set_name=e_set,
        )
        rows["E_set"] = e_set
        fold_detail_rows.append(rows)
        tuning_trace_rows.append(tuning_rows)
        sample_fold_rows.append(sample_rows)
        metric_rows.append(summarize_rows(rows, e_set=e_set))
        print(
            f"{e_set} done: "
            f"AUC={metric_rows[-1]['AUC_mean']:.4f}±{metric_rows[-1]['AUC_std']:.4f}"
        )

        # Incremental checkpoints for long-running experiments.
        pd.DataFrame(metric_rows).to_csv(out_dir / "metrics_table.csv", index=False)
        pd.concat(fold_detail_rows, ignore_index=True).to_csv(
            out_dir / "deepforest_spatial_fold_details.csv", index=False
        )
        pd.concat(tuning_trace_rows, ignore_index=True).to_csv(
            out_dir / "deepforest_tuning_trace.csv", index=False
        )
        pd.concat(sample_fold_rows, ignore_index=True).to_csv(
            out_dir / "deepforest_spatial_sample_folds.csv", index=False
        )

    metrics_table = pd.DataFrame(metric_rows)
    metrics_path = out_dir / "metrics_table.csv"
    metrics_table.to_csv(metrics_path, index=False)

    fold_details = pd.concat(fold_detail_rows, ignore_index=True)
    fold_path = out_dir / "deepforest_spatial_fold_details.csv"
    fold_details.to_csv(fold_path, index=False)
    tuning_trace = pd.concat(tuning_trace_rows, ignore_index=True)
    tuning_path = out_dir / "deepforest_tuning_trace.csv"
    tuning_trace.to_csv(tuning_path, index=False)
    sample_folds = pd.concat(sample_fold_rows, ignore_index=True)
    sample_folds_path = out_dir / "deepforest_spatial_sample_folds.csv"
    sample_folds.to_csv(sample_folds_path, index=False)
    print("Wrote:", metrics_path)
    print("Wrote:", fold_path)
    print("Wrote:", tuning_path)
    print("Wrote:", sample_folds_path)

    if args.run_random_e1:
        feats = feature_sets["E1"]
        work = df.dropna(subset=feats + [args.target_col]).copy()
        X = work[feats].values
        y = work[args.target_col].values
        rand_rows = nested_random_e1_eval(
            X=X,
            y=y,
            params_list=params_list,
            n_splits=args.n_blocks,
            seed=args.seed,
            patience_layers=args.patience_layers,
            delta_auc=args.delta_auc,
            threshold=args.threshold,
        )
        spat_rows = fold_details[fold_details["E_set"] == "E1"].copy()
        compare = pd.DataFrame(
            [
                {
                    "mode": "spatial",
                    **summarize_rows(spat_rows, e_set="E1"),
                },
                {
                    "mode": "random",
                    **summarize_rows(rand_rows, e_set="E1"),
                },
            ]
        )
        compare = compare.drop(columns=["E_set"])
        cmp_path = out_dir / "compare_random_vs_spatial_E1.csv"
        compare.to_csv(cmp_path, index=False)
        print("Wrote:", cmp_path)


if __name__ == "__main__":
    main()
