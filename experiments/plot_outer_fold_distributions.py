#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot outer-fold metric distributions by feature set and model.

Example:
  python experiments/plot_outer_fold_distributions.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make outer-fold boxplots from fold-level metrics."
    )
    parser.add_argument(
        "--input",
        default="results/tables/custom_table_stat_tests_95ci_wilcoxon_holm/custom_table_fold_level_data.csv",
        help="Fold-level metrics CSV path.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/figures/outer_fold_distributions_feature_model_custom_table",
        help="Output figure directory.",
    )
    parser.add_argument(
        "--show_fliers",
        action="store_true",
        help="Show boxplot outlier points. Hidden by default.",
    )
    return parser.parse_args()


def plot_metric(df, metric: str, ylabel: str, title: str, out_path: Path, show_fliers: bool) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    model_order = ["DeepForest", "RF", "XGB", "LGBM", "CAT"]
    feature_order = ["E1", "E2", "E3", "E4"]

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=100)
    sns.boxplot(
        data=df,
        x="E_set",
        y=metric,
        hue="Model",
        order=feature_order,
        hue_order=model_order,
        ax=ax,
        showfliers=show_fliers,
        width=0.75,
    )

    ax.set_title(title, fontsize=13, pad=9)
    ax.set_xlabel("Feature Set")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.3, 1.0)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit("Please install pandas before running this script.") from exc

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not input_path.exists():
        raise SystemExit(f"File not found: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    df = df.copy()
    df["Model"] = df["Model"].replace({"DF": "DeepForest"})

    plot_metric(
        df,
        metric="AUC",
        ylabel="AUC",
        title="Outer-Fold AUC Distribution by Feature Set and Model",
        out_path=out_dir / "outer_fold_auc_boxplot_reference_style_y03_10.png",
        show_fliers=args.show_fliers,
    )
    plot_metric(
        df,
        metric="F1",
        ylabel="F1-score",
        title="Outer-Fold F1-score Distribution by Feature Set and Model",
        out_path=out_dir / "outer_fold_f1_boxplot_reference_style_y03_10.png",
        show_fliers=args.show_fliers,
    )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
