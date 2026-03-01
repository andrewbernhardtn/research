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
    filter_eval_paths,
)


def main() -> None:
    # ---------------------------------------------------------------------
    # Paths (relative to *this* runner file):
    # research/
    #   data/
    #   results/
    #   script/  <-- this file lives here
    # ---------------------------------------------------------------------
    RESEARCH_DIR = Path(__file__).resolve().parents[1]  # .../research
    DATA_DIR = RESEARCH_DIR / "data"
    RESULTS_DIR = RESEARCH_DIR / "results_clean"

    # ---------------------------------------------------------------------
    # CONFIG (EDIT THESE)
    # ---------------------------------------------------------------------
    TRAIN_PATH = DATA_DIR / "lf_2021-11-25.parquet"

    DO_EVAL = False  # <-- False for training, True for validating other locations

    EVAL_MODE = "only_dates"  # "all" or "first_n" or "only_dates"
    EVAL_FIRST_N = 3
    EVAL_ONLY_DATES = {"2021-11-04", "2022-01-28"}  # used only if mode="only_dates"

    dataset = infer_dataset_from_filename(TRAIN_PATH)
    train_date = TRAIN_PATH.stem.split("_", 1)[1]  # "2021-11-13" from "hf_2021-11-13"

    eval_paths = discover_eval_paths(DATA_DIR, dataset, exclude=TRAIN_PATH)
    EVAL_PATHS = filter_eval_paths(eval_paths, EVAL_MODE, EVAL_FIRST_N, EVAL_ONLY_DATES)


    if DO_EVAL and not EVAL_PATHS:
        raise ValueError("DO_EVAL=True but EVAL_PATHS is empty. Add eval parquet datasets or set DO_EVAL=False.")


    if dataset == "hf":
        cfg = HDBSCANConfig(
            features=["attenuation_dB", "carrier_frequency_kHz"],
            freq_unit="MHz",
            min_cluster_size=5000,
            min_samples=1500,
            cluster_selection_epsilon=0.3,
            rolling_window=7,
            use_pca=False,
            cluster_selection_method="eom",
            metric="manhattan",
        )
    else:
        cfg = HDBSCANConfig(
            features=["attenuation_dB", "carrier_frequency_kHz"],
            freq_unit="kHz",
            min_cluster_size=8000,
            min_samples=1200,
            cluster_selection_epsilon=0.0,
            rolling_window=3,
            use_pca=False,
            cluster_selection_method="eom",
            metric="manhattan",
        )

    run_name = "train_lf_04"
    save_dir = ensure_dir(RESULTS_DIR / run_name)

    # ---------------------------------------------------------------------
    # 1) TRAIN (fit preprocess + fit HDBSCAN on reference day)
    # ---------------------------------------------------------------------
    preprocess, clusterer, train_plot_df, X_train, train_labels = fit_reference_model(
        train_path=TRAIN_PATH,
        cfg=cfg,
    )

    n_train_clusters, n_train_noise = cluster_counts(train_labels)
    train_sample_tag = sample_tag(cfg.sample_fraction, len(X_train))
    pca_str = pca_info(cfg)

    print(
        f"✅ TRAIN fitted: {TRAIN_PATH.stem} | {train_sample_tag} | "
        f"Clusters={n_train_clusters}, Noise={n_train_noise} | "
        f"mc={cfg.min_cluster_size}, ms={cfg.min_samples}, eps={cfg.cluster_selection_epsilon}, "
        f"method={cfg.cluster_selection_method}, metric={cfg.metric} {pca_str}"
    )

    train_title = (
        f"{TRAIN_PATH.stem} | "
        f"mc={cfg.min_cluster_size}, ms={cfg.min_samples}, eps={cfg.cluster_selection_epsilon} | "
        f"method={cfg.cluster_selection_method}, metric={cfg.metric}"
    )

    train_out = save_dir / (
        f"hdbscan_{dataset}_{train_date}_"
        f"mc{cfg.min_cluster_size}_"
        f"ms{cfg.min_samples}_"
        f"eps{int(cfg.cluster_selection_epsilon * 1000)}_"
        f"{cfg.cluster_selection_method}_"
        f"{cfg.metric}.png"
    )

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
    # 2) EVAL (apply to other HF days, no refit)
    # ---------------------------------------------------------------------
    if DO_EVAL:
        for eval_path in EVAL_PATHS:
            try:
                eval_plot_df, X_eval, pred_labels, pred_strengths = apply_model_to_day(
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
                f"✅ EVAL applied: {eval_path.stem} | {eval_sample_tag} | "
                f"PredClusters={n_eval_clusters}, PredNoise={n_eval_noise} ({pct_assigned:.1f}% assigned)"
            )

            eval_title = (
                f"{eval_path.stem} | "
                f"mc={cfg.min_cluster_size}, ms={cfg.min_samples}, eps={cfg.cluster_selection_epsilon} | "
                f"method={cfg.cluster_selection_method}, metric={cfg.metric}"
            )

            eval_date = eval_path.stem.split("_", 1)[1]

            eval_out = save_dir / (
                f"hdbscan_{dataset}_{eval_date}_"
                f"mc{cfg.min_cluster_size}_"
                f"ms{cfg.min_samples}_"
                f"eps{int(cfg.cluster_selection_epsilon * 1000)}_"
                f"{cfg.cluster_selection_method}_"
                f"{cfg.metric}.png"
            )

            plot_clusters(
                df_plot=eval_plot_df,
                X_index=X_eval.index,
                labels=pred_labels,
                title=eval_title,
                out_path=eval_out,
                cfg=cfg,
            )
            print(f"   ↳ saved eval plot: {eval_out.name}")
    else:
        print("Use DO_EVAL=True for evaluation on other days.")

if __name__ == "__main__":
    main()