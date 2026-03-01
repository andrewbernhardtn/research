#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from script.oop_functions import HDBSCANWorkflow


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
    RUN_MODE = "train_model"  # "train_model" | "validate_model"

    # hf vs lf is inferred from this filename ("hf_..." or "lf_...")
    TRAIN_PATH = DATA_DIR / "lf_2021-11-25.parquet"
    # TRAIN_PATH = DATA_DIR / "hf_2021-11-13.parquet"

    # Which days to validate on (only used if RUN_MODE="validate_model")
    EVAL_MODE = "dates"  # "all" | "dates"
    EVAL_DATES = {"2021-11-04", "2022-01-28"}  # used only if EVAL_MODE="dates"

    # Output folder name
    run_name = "test_oop_01"

    # ---------------------------------------------------------------------
    # RUN WORKFLOW
    # ---------------------------------------------------------------------
    wf = HDBSCANWorkflow(
        train_path=TRAIN_PATH,
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        run_name=run_name,
        run_mode=RUN_MODE,
        eval_mode=EVAL_MODE,
        eval_dates=set(EVAL_DATES),
        cfg=None,  # keep None to use default hf/lf config; or pass a custom HDBSCANConfig
    )
    wf.run()


if __name__ == "__main__":
    main()