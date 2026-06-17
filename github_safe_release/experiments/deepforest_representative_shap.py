#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepForest SHAP global plots for representative spatial outer folds.

Representative fold rule:
For each E-set, choose the outer fold whose AUC is closest to that E-set's
mean outer AUC from deepforest_spatial_fold_details.csv.

The original nested CV did not persist fitted models, so this script rebuilds
the same spatial splits, retrains the selected fold model with the selected
parameters, and computes model-agnostic SHAP values for positive-class
probabilities on a sampled outer-test subset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GroupKFold

from deepforest_e1e4_nested_spatial import (
    DEFAULT_NON_FEATURE_COLS,
    build_spatial_blocks,
    infer_feature_sets,
    make_model,
    parse_extra_excludes,
    predict_proba_at_layer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build representative-fold DeepForest SHAP global plots."
    )
    parser.add_argument(
        "--input_csv",
        default="data/processed/master_0.5_7_30_allrows.csv",
        help="Input processed CSV.",
    )
    parser.add_argument(
        "--fold_details_csv",
        default=(
            "results/tables/"
            "deepforest_e1e4_independent_tuning_0.5_7_30_full_date_light_n10/"
            "deepforest_spatial_fold_details.csv"
        ),
        help="DeepForest spatial fold details CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/figures/deepforest_representative_shap_light_n10",
        help="Output directory for SHAP figures and tables.",
    )
    parser.add_argument("--lon_col", default="lon_bin")
    parser.add_argument("--lat_col", default="lat_bin")
    parser.add_argument("--date_col", default="full_date")
    parser.add_argument("--target_col", default="label")
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude_cols",
        default="鲐鱼/t,date,latitude,longitude",
        help="Extra non-feature columns to exclude, comma-separated.",
    )
    parser.add_argument(
        "--background_n",
        type=int,
        default=100,
        help="Background rows sampled from outer train for model-agnostic SHAP.",
    )
    parser.add_argument(
        "--explain_n",
        type=int,
        default=300,
        help="Outer test rows sampled for SHAP explanations.",
    )
    parser.add_argument("--max_display", type=int, default=20)
    return parser.parse_args()


def choose_representative_folds(fold_details: pd.DataFrame):
    rows = []
    for e_set, group in fold_details.groupby("E_set", sort=True):
        mean_auc = float(group["test_auc"].mean())
        work = group.copy()
        work["abs_auc_diff"] = (work["test_auc"] - mean_auc).abs()
        best = work.sort_values(["abs_auc_diff", "outer_fold"]).iloc[0]
        rows.append(
            {
                "E_set": e_set,
                "mean_outer_auc": mean_auc,
                "representative_outer_fold": int(best["outer_fold"]),
                "representative_fold_auc": float(best["test_auc"]),
                "abs_auc_diff": float(best["abs_auc_diff"]),
                "selected_layer": int(best["selected_layer"]),
                "best_param": best["best_param"],
            }
        )
    return pd.DataFrame(rows)


def stratified_sample_frame(
    X_df: pd.DataFrame,
    y: np.ndarray,
    indices: np.ndarray,
    n: int,
    seed: int,
):
    pool = pd.DataFrame({"row_pos": indices, "y": y[indices]})
    if len(pool) <= n:
        chosen = pool["row_pos"].to_numpy()
        return X_df.iloc[chosen].copy(), {
            "n": int(len(pool)),
            "positive": int(pool["y"].sum()),
            "negative": int((1 - pool["y"]).sum()),
        }

    rng = np.random.default_rng(seed)
    classes = sorted(pool["y"].unique().tolist())
    if len(classes) < 2:
        chosen = pool.sample(n=n, random_state=seed)["row_pos"].to_numpy()
    else:
        n_pos_total = int((pool["y"] == 1).sum())
        n_neg_total = int((pool["y"] == 0).sum())
        n_pos = min(n_pos_total, n // 2)
        n_neg = min(n_neg_total, n - n_pos)
        shortfall = n - n_pos - n_neg
        if shortfall > 0:
            if n_pos_total - n_pos >= shortfall:
                n_pos += shortfall
            elif n_neg_total - n_neg >= shortfall:
                n_neg += shortfall

        pos_pool = pool[pool["y"] == 1]["row_pos"].to_numpy()
        neg_pool = pool[pool["y"] == 0]["row_pos"].to_numpy()
        chosen = np.concatenate(
            [
                rng.choice(pos_pool, size=n_pos, replace=False),
                rng.choice(neg_pool, size=n_neg, replace=False),
            ]
        )
        rng.shuffle(chosen)

    sampled_y = y[chosen]
    return X_df.iloc[chosen].sort_index().copy(), {
        "n": int(len(chosen)),
        "positive": int(sampled_y.sum()),
        "negative": int((1 - sampled_y).sum()),
    }


def train_representative_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_fold: int,
    params: dict,
    selected_layer: int,
    n_blocks: int,
    seed: int,
):
    outer_cv = GroupKFold(n_splits=n_blocks)
    for fold_id, (tr_idx, te_idx) in enumerate(
        outer_cv.split(X, y, groups=groups), start=1
    ):
        if fold_id == outer_fold:
            model = make_model(params, seed=seed)
            model.fit(X[tr_idx], y[tr_idx])
            layer = min(int(selected_layer), int(model.n_layers_))
            return model, layer, tr_idx, te_idx
    raise ValueError(f"outer_fold={outer_fold} not found")


def positive_class_predict_fn(model, selected_layer: int):
    def predict_fn(x):
        x_arr = np.asarray(x)
        return predict_proba_at_layer(model, x_arr, selected_layer)

    return predict_fn


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if "sample_id" not in df.columns:
        df = df.copy()
        df["sample_id"] = np.arange(len(df), dtype=int)
    df["spatial_block"] = build_spatial_blocks(
        df=df,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        n_blocks=args.n_blocks,
    )

    drop_cols = {
        args.lon_col,
        args.lat_col,
        args.date_col,
        args.target_col,
        *DEFAULT_NON_FEATURE_COLS,
        *parse_extra_excludes(args.exclude_cols),
    }
    feature_sets = infer_feature_sets(df, drop_cols=drop_cols)

    fold_details = pd.read_csv(args.fold_details_csv)
    representatives = choose_representative_folds(fold_details)
    representatives.to_csv(out_dir / "representative_folds.csv", index=False)
    print(representatives.to_string(index=False), flush=True)

    importance_rows = []

    for _, rep in representatives.iterrows():
        e_set = rep["E_set"]
        outer_fold = int(rep["representative_outer_fold"])
        selected_layer = int(rep["selected_layer"])
        params = json.loads(rep["best_param"])
        features = feature_sets[e_set]
        work = df.dropna(subset=features + [args.target_col]).copy()
        X_df = work[features].copy()
        X = X_df.values
        y = work[args.target_col].astype(int).values
        groups = work["spatial_block"].values

        model, layer, tr_idx, te_idx = train_representative_model(
            X=X,
            y=y,
            groups=groups,
            outer_fold=outer_fold,
            params=params,
            selected_layer=selected_layer,
            n_blocks=args.n_blocks,
            seed=args.seed,
        )

        train_df = X_df.iloc[tr_idx]
        test_df = X_df.iloc[te_idx]
        background, background_counts = stratified_sample_frame(
            X_df=X_df,
            y=y,
            indices=tr_idx,
            n=args.background_n,
            seed=args.seed + outer_fold,
        )
        explain_df, explain_counts = stratified_sample_frame(
            X_df=X_df,
            y=y,
            indices=te_idx,
            n=args.explain_n,
            seed=args.seed + 100 + outer_fold,
        )

        print(
            f"[{e_set}] representative_fold={outer_fold}, "
            f"test_auc={rep['representative_fold_auc']:.4f}, "
            f"n_train={len(train_df)}, n_test={len(test_df)}, "
            f"n_background={len(background)} "
            f"(pos={background_counts['positive']}, neg={background_counts['negative']}), "
            f"n_explain={len(explain_df)} "
            f"(pos={explain_counts['positive']}, neg={explain_counts['negative']}), "
            f"layer={layer}, params={params}",
            flush=True,
        )
        pd.DataFrame(
            [
                {
                    "E_set": e_set,
                    "outer_fold": outer_fold,
                    "sample": "background",
                    **background_counts,
                },
                {
                    "E_set": e_set,
                    "outer_fold": outer_fold,
                    "sample": "explain",
                    **explain_counts,
                },
            ]
        ).to_csv(out_dir / f"{e_set}_stratified_sample_counts.csv", index=False)

        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(
            positive_class_predict_fn(model, layer),
            masker,
            algorithm="permutation",
            seed=args.seed,
        )
        max_evals = max(2 * len(features) + 1, 50)
        explanation = explainer(explain_df, max_evals=max_evals, batch_size=64)

        shap_values_path = out_dir / f"{e_set}_representative_shap_values.csv"
        shap_df = pd.DataFrame(
            explanation.values,
            columns=features,
            index=explain_df.index,
        )
        shap_df.insert(0, "sample_index", explain_df.index.astype(int))
        shap_df.to_csv(shap_values_path, index=False)

        mean_abs = np.abs(explanation.values).mean(axis=0)
        imp = pd.DataFrame(
            {
                "E_set": e_set,
                "outer_fold": outer_fold,
                "feature": features,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(out_dir / f"{e_set}_mean_abs_shap.csv", index=False)
        importance_rows.extend(imp.to_dict("records"))

        plt.figure()
        shap.plots.beeswarm(
            explanation,
            max_display=args.max_display,
            show=False,
            color_bar=True,
        )
        plt.title(f"{e_set} representative fold {outer_fold} SHAP summary")
        plt.tight_layout()
        plt.savefig(out_dir / f"{e_set}_shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.plots.bar(explanation, max_display=args.max_display, show=False)
        plt.title(f"{e_set} representative fold {outer_fold} mean |SHAP|")
        plt.tight_layout()
        plt.savefig(out_dir / f"{e_set}_shap_bar.png", dpi=300, bbox_inches="tight")
        plt.close()

    pd.DataFrame(importance_rows).to_csv(
        out_dir / "all_representative_mean_abs_shap.csv", index=False
    )
    print("Wrote:", out_dir, flush=True)


if __name__ == "__main__":
    main()
