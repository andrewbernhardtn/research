#!/usr/bin/env python3
"""
load_model.py
=============
Load a pre-trained HDBSCAN model and apply it to new data.

This script is intended for engineers who have received the following
three files and want to run predictions without retraining:

    hdbscan_model.py        ← class definitions (required)
    preprocessor.joblib     ← fitted StandardScaler
    cluster_model.joblib    ← fitted HDBSCAN model

Usage
-----
1. Place this script in the same folder as hdbscan_model.py.
2. Edit the USER CONTROLS section below.
3. Run:  python load_model.py

Output
------
A scatter plot (PNG) saved to the folder specified by OUTPUT_DIR,
showing the new data coloured by predicted cluster label.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hdbscan_model import (
    Preprocessor,
    ClusterModel,
    ClusterVisualizer,
    HDBSCANConfig,
    clean_features,
    sample_df,
    sample_tag,
)


# ---------------------------------------------------------------------------
# Logging  — change level to DEBUG for verbose output
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # -----------------------------------------------------------------------
    # Paths  (resolved relative to this file, so you can run from anywhere)
    # -----------------------------------------------------------------------
    SCRIPT_DIR   = Path(__file__).resolve().parent       # .../research/script
    RESEARCH_DIR = SCRIPT_DIR.parent                     # .../research

    # -----------------------------------------------------------------------
    # USER CONTROLS  ← edit these before running
    # -----------------------------------------------------------------------

    # Folder that contains preprocessor.joblib and cluster_model.joblib
    # (the folder shared with you by the model owner)
    MODEL_DIR: Path = RESEARCH_DIR / "final_results" / "experiment_03"

    # Path to the new parquet file you want to predict on
    NEW_DATA_PATH: Path = RESEARCH_DIR / "data" / "hf_2021-11-05.parquet"

    # Folder where the output plot will be saved
    OUTPUT_DIR: Path = RESEARCH_DIR / "final_results" / "experiment_03"

    # Features to use — must match what the model was trained on
    FEATURES: list[str] = ["attenuation_dB", "carrier_frequency_kHz"]

    # Sampling — set to None to use all rows
    SAMPLE_FRACTION: float | None = 0.01
    SAMPLE_RANDOM_STATE: int      = 42

    # -----------------------------------------------------------------------
    # Load the saved model
    # -----------------------------------------------------------------------
    logger.info("LOAD  | loading preprocessor and cluster model from %s", MODEL_DIR.name)

    preprocessor = Preprocessor.load(MODEL_DIR / "preprocessor.joblib")
    model        = ClusterModel.load(MODEL_DIR / "cluster_model.joblib")

    logger.info("LOAD  | preprocessor loaded ✓")
    logger.info("LOAD  | cluster model loaded ✓  params: %s", model.param_summary)

    # -----------------------------------------------------------------------
    # Load and prepare new data
    # -----------------------------------------------------------------------
    logger.info("DATA  | reading %s", NEW_DATA_PATH.name)

    df         = pd.read_parquet(NEW_DATA_PATH, columns=FEATURES)
    df_sampled = sample_df(df, SAMPLE_FRACTION, SAMPLE_RANDOM_STATE)
    X          = clean_features(df_sampled, FEATURES)

    if X.empty:
        raise RuntimeError(f"No usable rows after cleaning: {NEW_DATA_PATH}")

    tag = sample_tag(SAMPLE_FRACTION, len(X))
    logger.info("DATA  | %s rows ready for prediction", tag)

    # -----------------------------------------------------------------------
    # Transform and predict
    # -----------------------------------------------------------------------
    X_transformed      = preprocessor.transform(X)
    labels, strengths  = model.predict(X_transformed)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(labels == -1))
    pct_assigned = 100.0 * (1.0 - n_noise / max(len(labels), 1))

    logger.info(
        "PRED  | clusters=%d, noise=%d, assigned=%.1f%%",
        n_clusters, n_noise, pct_assigned,
    )

    # -----------------------------------------------------------------------
    # Visualise and save plot
    # -----------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = model.cfg
    filename = (
        f"loaded_hdbscan_{NEW_DATA_PATH.stem}_"
        f"mc{cfg.min_cluster_size}_"
        f"ms{cfg.min_samples}_"
        f"eps{int(cfg.cluster_selection_epsilon * 1000)}_"
        f"{cfg.cluster_selection_method}_"
        f"{cfg.metric}.png"
    )
    out_path = OUTPUT_DIR / filename
    title    = f"{NEW_DATA_PATH.stem} [loaded model] | {model.param_summary}"

    plot_df = df_sampled.loc[X.index]

    visualizer = ClusterVisualizer(cfg)
    visualizer.plot(
        df_plot=plot_df,
        X_index=X.index,
        labels=labels,
        title=title,
        out_path=out_path,
    )
    logger.info("PLOT  | saved → %s", out_path.name)


if __name__ == "__main__":
    main()
