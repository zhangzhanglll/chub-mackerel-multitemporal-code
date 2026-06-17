#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RF/XGB/LGBM/CAT under Nested Spatial CV for E1-E4.

Outputs:
- baseline_nested_metrics_table.csv
- baseline_nested_fold_details.csv
- baseline_nested_tuning_trace.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


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

MODEL_LIST = ["RF", "XGB", "LGBM", "CAT"]


def parse_args():
    p = argparse.ArgumentParser(description="Tree baselines nested spatial CV for E1-E4")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--out_dir", default="results/tables/nested_spatial_medium_v2")
    p.add_argument("--lon_col", default="lon_bin")
    p.add_argument("--lat_col", default="lat_bin")
    p.add_argument("--date_col", default="date")
    p.add_argument("--target_col", default="label")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_blocks", type=int, default=6)
    p.add_argument("--n_candidates", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--light_profile",
        action="store_true",
        help="Use smaller parameter grids for full model x E1-E4 reruns on large data.",
    )
    p.add_argument(
        "--exclude_cols",
        default="",
        help="Extra non-feature columns to exclude, comma-separated.",
    )
    p.add_argument(
        "--models",
        default="RF,XGB,LGBM,CAT",
        help="Comma-separated models in RF,XGB,LGBM,CAT",
    )
    return p.parse_args()


def safe_auc(y_true, prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, prob))


def metrics_from_prob(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    auc = safe_auc(y_true, prob)
    if np.isnan(auc):
        auc = 0.5
    return {
        "auc": float(auc),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)),
    }


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


def build_spatial_blocks(df: pd.DataFrame, lon_col: str, lat_col: str, n_blocks: int):
    coords = df[[lon_col, lat_col]].values
    km = KMeans(n_clusters=n_blocks, random_state=42, n_init=10)
    return km.fit_predict(coords)


def build_param_space(model_name: str, light_profile: bool = False):
    if model_name == "RF":
        if light_profile:
            return {
                "n_estimators": [50, 100],
                "max_depth": [8, 12, None],
                "min_samples_leaf": [3, 5],
                "max_features": ["sqrt"],
            }
        return {
            "n_estimators": [300, 500],
            "max_depth": [None, 15],
            "min_samples_leaf": [1, 3],
            "max_features": ["sqrt", 0.7],
        }
    if model_name == "XGB":
        if light_profile:
            return {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            }
        return {
            "n_estimators": [300, 500],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
        }
    if model_name == "LGBM":
        if light_profile:
            return {
                "n_estimators": [100, 200],
                "num_leaves": [31, 63],
                "learning_rate": [0.03, 0.1],
                "feature_fraction": [0.8],
                "bagging_fraction": [0.8],
            }
        return {
            "n_estimators": [300, 500],
            "num_leaves": [31, 63],
            "learning_rate": [0.03, 0.1],
            "feature_fraction": [0.7, 1.0],
            "bagging_fraction": [0.7, 1.0],
        }
    if model_name == "CAT":
        if light_profile:
            return {
                "iterations": [200, 400],
                "depth": [4, 6],
                "learning_rate": [0.03, 0.1],
                "l2_leaf_reg": [3],
            }
        return {
            "iterations": [400, 800],
            "depth": [4, 6],
            "learning_rate": [0.03, 0.1],
            "l2_leaf_reg": [3, 9],
        }
    raise ValueError(f"Unsupported model: {model_name}")


def sample_param_candidates(grid: dict, n_candidates: int, seed: int):
    keys = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    if n_candidates >= len(combos):
        return combos
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(combos), size=n_candidates, replace=False)
    return [combos[i] for i in idx]


def make_model(model_name: str, params: dict, seed: int):
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "XGB":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed")
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=seed,
            n_jobs=-1,
            eval_metric="auc",
            use_label_encoder=False,
        )
    if model_name == "LGBM":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed")
        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            feature_fraction=params["feature_fraction"],
            bagging_fraction=params["bagging_fraction"],
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if model_name == "CAT":
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed")
        return CatBoostClassifier(
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params["l2_leaf_reg"],
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def nested_spatial_cv_eval(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_meta: pd.DataFrame,
    params_list: list[dict],
    n_blocks: int,
    seed: int,
    threshold: float,
    e_set_name: str,
):
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
        print(f"[{model_name}][{e_set_name}][outer_fold={outer_fold}] len(intersection)=={len(intersection)}")
        assert len(intersection) == 0

        best_param = None
        best_inner_auc = -np.inf

        n_unique = len(np.unique(g_tr))
        inner_splits = min(n_blocks - 1, n_unique)
        if inner_splits < 2:
            raise ValueError(f"outer_fold={outer_fold}: inner_splits={inner_splits} < 2")
        inner_cv = GroupKFold(n_splits=inner_splits)

        for param_idx, params in enumerate(params_list, start=1):
            print(
                f"[{model_name}][{e_set_name}][outer_fold={outer_fold}] "
                f"candidate {param_idx}/{len(params_list)} start, params={params}",
                flush=True,
            )
            aucs = []
            for in_tr_idx, in_va_idx in inner_cv.split(X_tr, y_tr, groups=g_tr):
                X_in_tr, y_in_tr = X_tr[in_tr_idx], y_tr[in_tr_idx]
                X_in_va, y_in_va = X_tr[in_va_idx], y_tr[in_va_idx]

                model = make_model(model_name, params, seed=seed)
                model.fit(X_in_tr, y_in_tr)
                prob = model.predict_proba(X_in_va)[:, 1]
                auc = safe_auc(y_in_va, prob)
                if not np.isnan(auc):
                    aucs.append(auc)

            mean_auc = float(np.mean(aucs)) if aucs else float("nan")
            tuning_rows.append(
                {
                    "model": model_name,
                    "E_set": e_set_name,
                    "outer_fold": outer_fold,
                    "candidate_idx": param_idx,
                    "n_candidates": len(params_list),
                    "mean_inner_auc": mean_auc,
                    "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                }
            )
            print(
                f"[{model_name}][{e_set_name}][outer_fold={outer_fold}] "
                f"candidate {param_idx}/{len(params_list)} mean_inner_auc={mean_auc:.4f}, params={params}",
                flush=True,
            )
            if not np.isnan(mean_auc) and mean_auc > best_inner_auc:
                best_inner_auc = mean_auc
                best_param = params

        if best_param is None:
            best_param = params_list[0]
            best_inner_auc = 0.5
        print(
            f"[{model_name}][{e_set_name}][outer_fold={outer_fold}] "
            f"selected mean_inner_auc={best_inner_auc:.4f}, best_param={best_param}",
            flush=True,
        )

        model = make_model(model_name, best_param, seed=seed)
        model.fit(X_tr, y_tr)
        prob_te = model.predict_proba(X_te)[:, 1]
        m = metrics_from_prob(y_te, prob_te, threshold=threshold)
        rows.append(
            {
                "model": model_name,
                "E_set": e_set_name,
                "outer_fold": outer_fold,
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
                    "model": model_name,
                    "E_set": e_set_name,
                    "sample_id": meta_row["sample_id"],
                    "grid_id": meta_row.get("grid_id", ""),
                    "spatial_block": int(meta_row["spatial_block"]),
                    "fold_id": outer_fold,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(tuning_rows), pd.DataFrame(sample_fold_rows)


def summarize_rows(rows: pd.DataFrame, model_name: str, e_set: str):
    return {
        "model": model_name,
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


def parse_extra_excludes(exclude_cols_arg: str) -> set[str]:
    if not exclude_cols_arg.strip():
        return set()
    return {c.strip() for c in exclude_cols_arg.split(",") if c.strip()}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_names:
        if m not in MODEL_LIST:
            raise SystemExit(f"Unsupported model in --models: {m}")

    df = pd.read_csv(args.input_csv)
    required = {args.lon_col, args.lat_col, args.target_col}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    if "sample_id" not in df.columns:
        df["sample_id"] = np.arange(len(df), dtype=int)
    df["spatial_block"] = build_spatial_blocks(
        df=df, lon_col=args.lon_col, lat_col=args.lat_col, n_blocks=args.n_blocks
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

    metric_rows = []
    fold_detail_rows = []
    tuning_trace_rows = []
    sample_fold_rows = []

    for model_name in model_names:
        params_list = sample_param_candidates(
            build_param_space(model_name, light_profile=args.light_profile),
            n_candidates=args.n_candidates,
            seed=args.seed,
        )
        for e_set in ["E1", "E2", "E3", "E4"]:
            feats = feature_sets[e_set]
            work = df.dropna(subset=feats + [args.target_col]).copy()
            X = work[feats].values
            y = work[args.target_col].values
            g = work["spatial_block"].values
            sample_meta = work[["sample_id", "grid_id", "spatial_block"]].copy()
            if len(np.unique(g)) < args.n_blocks:
                raise SystemExit(
                    f"{model_name}-{e_set}: unique blocks < n_blocks "
                    f"have={len(np.unique(g))}, need={args.n_blocks}"
                )

            rows, tuning_rows, sample_rows = nested_spatial_cv_eval(
                model_name=model_name,
                X=X,
                y=y,
                groups=g,
                sample_meta=sample_meta,
                params_list=params_list,
                n_blocks=args.n_blocks,
                seed=args.seed,
                threshold=args.threshold,
                e_set_name=e_set,
            )
            fold_detail_rows.append(rows)
            tuning_trace_rows.append(tuning_rows)
            sample_fold_rows.append(sample_rows)
            metric_rows.append(summarize_rows(rows, model_name=model_name, e_set=e_set))
            print(
                f"{model_name}-{e_set} done: "
                f"AUC={metric_rows[-1]['AUC_mean']:.4f}±{metric_rows[-1]['AUC_std']:.4f}"
            )

            pd.DataFrame(metric_rows).to_csv(
                out_dir / "baseline_nested_metrics_table.csv", index=False
            )
            pd.concat(fold_detail_rows, ignore_index=True).to_csv(
                out_dir / "baseline_nested_fold_details.csv", index=False
            )
            pd.concat(tuning_trace_rows, ignore_index=True).to_csv(
                out_dir / "baseline_nested_tuning_trace.csv", index=False
            )
            pd.concat(sample_fold_rows, ignore_index=True).to_csv(
                out_dir / "baseline_nested_sample_folds.csv", index=False
            )

    print("Wrote:", out_dir / "baseline_nested_metrics_table.csv")
    print("Wrote:", out_dir / "baseline_nested_fold_details.csv")
    print("Wrote:", out_dir / "baseline_nested_tuning_trace.csv")
    print("Wrote:", out_dir / "baseline_nested_sample_folds.csv")


if __name__ == "__main__":
    main()
