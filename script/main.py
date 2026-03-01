#!/usr/bin/env python3
"""
main.py
=======
Entry point for the HDBSCAN clustering workflow.

Edit the USER CONTROLS section below to configure each run.
Everything else is handled automatically by HDBSCANWorkflow.

Project layout (relative to this file):
    research/
    ├── data/               ← parquet files (hf_*.parquet / lf_*.parquet)
    ├── results_clean/      ← output plots are saved here
    └── script/
        ├── main.py         ← YOU ARE HERE
        └── hdbscan_model.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from hdbscan_model import HDBSCANConfig, HDBSCANWorkflow, RunMode, EvalMode


# ---------------------------------------------------------------------------
# Logging  — change level to DEBUG for verbose output
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    # -----------------------------------------------------------------------
    # Paths  (resolved relative to this file, so you can run from anywhere)
    # -----------------------------------------------------------------------
    SCRIPT_DIR   = Path(__file__).resolve().parent       # .../research/script
    RESEARCH_DIR = SCRIPT_DIR.parent                     # .../research
    DATA_DIR     = RESEARCH_DIR / "data"
    RESULTS_DIR  = RESEARCH_DIR / "final_results"

    # -----------------------------------------------------------------------
    # USER CONTROLS  ← edit these for each experiment
    # -----------------------------------------------------------------------

    # "train_model"    → train + save plot only
    # "validate_model" → train, then predict on eval files
    RUN_MODE: RunMode = "validate_model"

    # Dataset type (hf / lf) is inferred automatically from the filename prefix
    # TRAIN_PATH = DATA_DIR / "lf_2021-11-25.parquet"
    TRAIN_PATH = DATA_DIR / "hf_2021-11-13.parquet"

    # Evaluation settings (only used when RUN_MODE = "validate_model")
    EVAL_MODE: EvalMode = "dates"                             # "all" | "dates"
    EVAL_DATES: set[str] = {"2021-11-05", "2022-01-21"} # ISO dates to evaluate

    # Output subfolder name inside RESULTS_DIR
    OUTPUT_FOLDER: str = "experiment_03"

    # Custom config — set to None to use the dataset default (recommended)
    # To override individual parameters, supply a full HDBSCANConfig, e.g.:
    #
    #   CUSTOM_CFG = HDBSCANConfig(
    #       features=["attenuation_dB", "carrier_frequency_kHz"],
    #       min_cluster_size=6_000,
    #       min_samples=1_000,
    #       cluster_selection_epsilon=0.2,
    #       cluster_selection_method="eom",
    #       metric="manhattan",
    #       freq_unit="kHz",
    #   )
    CUSTOM_CFG: HDBSCANConfig | None = None

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    wf = HDBSCANWorkflow(
        train_path=TRAIN_PATH,
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        output_folder=OUTPUT_FOLDER,
        run_mode=RUN_MODE,
        eval_mode=EVAL_MODE,
        eval_dates=EVAL_DATES,
        cfg=CUSTOM_CFG,
    )
    wf.run()


if __name__ == "__main__":
    main()
