#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Literal

from script.functions import (
    HDBSCANConfig,
    ensure_dir,
    fit_reference_model,
    apply_model_to_day,
    plot_clusters,
    cluster_counts,
    sample_tag,
    pca_info,
    infer_dataset_from_filename,
    discover_eval_paths,
    default_hdbscan_cfg,
    build_plot_out_path,
    filter_eval_paths
)

RunMode = Literal["train_model", "validate_model"]
EvalMode = Literal["all", "dates"]


def main() -> None:
    # ---------------------------------------------------------------------
    # Paths (relative to *this* runner file):
    # research/
    #   data/
    #   results_clean/
    #   script/  <-- this file lives here
    # ---------------------------------------------------------------------
    RESEARCH_DIR = Path(__file__).resolve().parents[1]  # .../research
    DATA_DIR = RESEARCH_DIR / "data"
    RESULTS_DIR = RESEARCH_DIR / "results_clean"

    # ---------------------------------------------------------------------
    # USER CONTROLS (EDIT THESE)
    # ---------------------------------------------------------------------
    RUN_MODE: RunMode = "validate_model"  # "train_model" | "validate_model"

    # hf vs lf is inferred from this filename ("hf_..." or "lf_...")
    TRAIN_PATH = DATA_DIR / "lf_2021-11-25.parquet"

    # Which days to validate on (only used if RUN_MODE="validate_model")
    EVAL_MODE: EvalMode = "dates"  # "all" | "dates"
    EVAL_DATES = {"2021-11-04", "2022-01-28"}  # used only if EVAL_MODE="dates"

    # Output folder name
    run_name = "train_lf_04"
    save_dir = ensure_dir(RESULTS_DIR / run_name)

    # ---------------------------------------------------------------------
    # AUTO: dataset + cfg + eval path discovery
    # ---------------------------------------------------------------------
    dataset = infer_dataset_from_filename(TRAIN_PATH)
    cfg = default_hdbscan_cfg(dataset)

    train_date = _date_from_stem(TRAIN_PATH)

    all_eval_paths = discover_eval_paths(DATA_DIR, dataset, exclude=TRAIN_PATH)
    EVAL_PATHS = filter_eval_paths(all_eval_paths, mode=EVAL_MODE, dates=set(EVAL_DATES))

    if RUN_MODE == "validate_model" and not EVAL_PATHS:
        raise ValueError(
            "RUN_MODE='validate_model' but no evaluation files were selected.\n"
            "• If EVAL_MODE='dates': check EVAL_DATES matches available parquet dates.\n"
            "• If EVAL_MODE='all': ensure there are other parquet files besides TRAIN_PATH."
        )

    # ---------------------------------------------------------------------
    # 1) TRAIN MODEL (always run in both modes)
    # ---------------------------------------------------------------------
    preprocess, clusterer, train_plot_df, X_train, train_labels = fit_reference_model(
        train_path=TRAIN_PATH,
        cfg=cfg,
    )

    n_train_clusters, n_train_noise = cluster_counts(train_labels)
    train_sample_tag = sample_tag(cfg.sample_fraction, len(X_train))
    pca_str = pca_info(cfg)

    print(
        f"✅ TRAIN_MODEL fitted: {TRAIN_PATH.stem} | {train_sample_tag} | "
        f"Clusters={n_train_clusters}, Noise={n_train_noise} | "
        f"mc={cfg.min_cluster_size}, ms={cfg.min_samples}, eps={cfg.cluster_selection_epsilon}, "
        f"method={cfg.cluster_selection_method}, metric={cfg.metric} {pca_str}"
    )

    train_title = (
        f"{TRAIN_PATH.stem} | "
        f"mc={cfg.min_cluster_size}, ms={cfg.min_samples}, eps={cfg.cluster_selection_epsilon} | "
        f"method={cfg.cluster_selection_method}, metric={cfg.metric}"
    )

    train_out = build_plot_out_path(save_dir, dataset=dataset, date_str=train_date, cfg=cfg)

    plot_clusters(
        df_plot=train_plot_df,
        X_index=X_train.index,
        labels=train_labels,
        title=train_title,
        out_path=train_out,
        cfg=cfg,
    )
    print(f"   ↳ saved train plot: {train_out.name}")

    # ---------------------------------------------------------------------
    # 2) VALIDATE MODEL (optional)
    # ---------------------------------------------------------------------
    if RUN_MODE == "train_model":
        return

    if RUN_MODE == "validate_model":
        for eval_path in EVAL_PATHS:
            try:
                eval_plot_df, X_eval, pred_labels, _pred_strengths = apply_model_to_day(
                    eval_path=eval_path,
                    cfg=cfg,
                    preprocess=preprocess,
                    clusterer=clusterer,
                )
            except RuntimeError as e:
                print(f"⚠️  {eval_path.stem}: {e} | skipping.")
                continue

            n_eval_clusters, n_eval_noise = cluster_counts(pred_labels)
            pct_assigned = 100.0 * (1.0 - (n_eval_noise / max(len(pred_labels), 1)))
            eval_sample_tag = sample_tag(cfg.sample_fraction, len(X_eval))

            print(
                f"✅ VALIDATE_MODEL applied: {eval_path.stem} | {eval_sample_tag} | "
                f"PredClusters={n_eval_clusters}, PredNoise={n_eval_noise} ({pct_assigned:.1f}% assigned)"
            )

            eval_title = (
                f"{eval_path.stem} | "
                f"mc={cfg.min_cluster_size}, ms={cfg.min_samples}, eps={cfg.cluster_selection_epsilon} | "
                f"method={cfg.cluster_selection_method}, metric={cfg.metric}"
            )

            eval_date = _date_from_stem(eval_path)
            eval_out = build_plot_out_path(save_dir, dataset=dataset, date_str=eval_date, cfg=cfg)

            plot_clusters(
                df_plot=eval_plot_df,
                X_index=X_eval.index,
                labels=pred_labels,
                title=eval_title,
                out_path=eval_out,
                cfg=cfg,
            )
            print(f"   ↳ saved eval plot: {eval_out.name}")
        return

    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()