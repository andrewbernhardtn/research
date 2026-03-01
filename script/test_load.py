#!/usr/bin/env python3
"""
test_load.py
============
Quick sanity check — verifies that the saved Preprocessor and ClusterModel
can be loaded from disk and are functional.
"""
from pathlib import Path
import numpy as np
from hdbscan_model import Preprocessor, ClusterModel

# --- Point this to your saved experiment folder ---
SAVE_DIR = Path(__file__).resolve().parent.parent / "final_results" / "experiment_03"

# 1. Load both objects
preprocessor = Preprocessor.load(SAVE_DIR / "preprocessor.joblib")
model        = ClusterModel.load(SAVE_DIR / "cluster_model.joblib")
print("✓ Both objects loaded successfully")

# 2. Check the preprocessor has a fitted scaler
print(f"  Scaler mean_  : {preprocessor._scaler.mean_}")
print(f"  Scaler scale_ : {preprocessor._scaler.scale_}")

# 3. Check the model has its fitted state
print(f"  Config        : {model.cfg}")
print(f"  Labels found  : {len(set(model.labels_))} unique (incl. noise=-1)")
print(f"  Rel. validity : {model.relative_validity_}")

# 4. Do a quick round-trip: create dummy data, transform, predict
dummy = np.array([[0.0, 500.0], [1.0, 800.0]], dtype="float32")
import pandas as pd
dummy_df = pd.DataFrame(dummy, columns=["attenuation_dB", "carrier_frequency_kHz"])
X_transformed = preprocessor.transform(dummy_df)
labels, strengths = model.predict(X_transformed)
print(f"  Dummy labels  : {labels}  (−1 = noise, expected for out-of-distribution points)")
print(f"  Strengths     : {strengths}")
print("✓ Round-trip transform → predict works")