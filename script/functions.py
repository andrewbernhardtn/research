from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Literal, Sequence, Dict, Any, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import hdbscan

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


Dataset = Literal["hf", "lf"]

def infer_dataset_from_filename(path: Path) -> Dataset:
    name = path.name.lower()
    if name.startswith("hf_"):
        return "hf"
    if name.startswith("lf_"):
        return "lf"
    raise ValueError(f"Cannot infer dataset type from filename: {path.name}. Expected prefix hf_ or lf_")

def discover_eval_paths(data_dir: Path, dataset: Dataset, exclude: Path) -> list[Path]:
    pattern = f"{dataset}_*.parquet"
    files = sorted(data_dir.glob(pattern))
    return [p for p in files if p.resolve() != exclude.resolve()]


EvalMode = Literal["all", "dates"]

def filter_eval_paths(paths: list[Path], mode: EvalMode, dates: set[str]) -> list[Path]:
    """
    Select evaluation parquet files.
    mode="all"   -> all discovered paths
    mode="dates" -> only those whose stem contains a date in `dates`
    """
    paths = sorted(paths, key=lambda p: p.name)

    if mode == "all":
        return paths

    if mode == "dates":
        return [p for p in paths if p.stem.split("_", 1)[1] in dates]

    raise ValueError("EVAL_MODE must be one of: all | dates")


@dataclass(frozen=True)
class HDBSCANConfig:
    # Features used for clustering / plotting
    features: list[str]
    freq_unit: str = "kHz"  # "kHz" or "MHz"

    # Preprocess
    use_pca: bool = False
    pca_n_components: int = 2

    # HDBSCAN
    min_cluster_size: int = 5000
    min_samples: int = 1500
    cluster_selection_epsilon: float = 0.3
    cluster_selection_method: str = "eom"
    metric: str = "manhattan"

    # Sampling
    sample_fraction: Optional[float] = 0.01
    sample_random_state: int = 42

    # Plotting
    fixed_ylim: tuple[float, float] = (-80.0, 100.0)
    rolling_window: int = 7


def default_hdbscan_cfg(dataset: Dataset) -> HDBSCANConfig:
    """
    Central place for hf/lf default hyperparameters.
    Keeps run_hdbscan.py clean and ensures consistency across scripts.
    """
    if dataset == "hf":
        return HDBSCANConfig(
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

    # lf defaults
    return HDBSCANConfig(
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


def build_plot_out_path(
    save_dir: Path,
    *,
    dataset: Dataset,
    date_str: str,
    cfg: HDBSCANConfig,
) -> Path:
    """
    Standardized plot filename builder so all scripts save consistent outputs.
    """
    return save_dir / (
        f"hdbscan_{dataset}_{date_str}_"
        f"mc{cfg.min_cluster_size}_"
        f"ms{cfg.min_samples}_"
        f"eps{int(cfg.cluster_selection_epsilon * 1000)}_"
        f"{cfg.cluster_selection_method}_"
        f"{cfg.metric}.png"
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_preprocess_pipeline(cfg: HDBSCANConfig) -> Pipeline:
    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
    if cfg.use_pca:
        steps.append(
            ("pca", PCA(n_components=cfg.pca_n_components, svd_solver="randomized", random_state=42))
        )
    return Pipeline(steps)


def sample_df(df: pd.DataFrame, frac: Optional[float], random_state: int) -> pd.DataFrame:
    if frac is None or frac >= 1:
        return df
    if frac <= 0:
        raise ValueError("sample_fraction must be > 0, or use None/1.0 for no sampling.")
    return df.sample(frac=frac, random_state=random_state)


def load_day(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")

    df = pd.read_parquet(path, columns=cols)

    # Keep dtypes consistent and compact
    if any(df[c].dtype != "float32" for c in cols):
        df[cols] = df[cols].astype("float32", copy=False)

    return df


def clean_feature_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    return X


def sample_tag(frac: Optional[float], n: int) -> str:
    if frac is None or frac >= 1:
        return f"full_n{n}"
    return f"s{frac:g}_n{n}"


def pca_info(cfg: HDBSCANConfig) -> str:
    return f"PCA({cfg.pca_n_components}D)" if cfg.use_pca else ""


def fit_reference_model(
    *,
    train_path: Path,
    cfg: HDBSCANConfig,
) -> tuple[Pipeline, hdbscan.HDBSCAN, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Loads train day, samples, cleans, fits preprocess + HDBSCAN.
    Returns:
      preprocess, clusterer, train_plot_df, X_train, train_labels
    """
    train_df = load_day(train_path, cfg.features)
    train_df_s = sample_df(train_df, cfg.sample_fraction, cfg.sample_random_state)
    X_train = clean_feature_frame(train_df_s, cfg.features)

    if len(X_train) == 0:
        raise RuntimeError(f"No usable rows in TRAIN_PATH after cleaning: {train_path}")

    train_plot_df = train_df_s.loc[X_train.index]

    preprocess = build_preprocess_pipeline(cfg)
    preprocess.fit(X_train)
    X_train_final = preprocess.transform(X_train)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(cfg.min_cluster_size),
        min_samples=int(cfg.min_samples),
        cluster_selection_epsilon=float(cfg.cluster_selection_epsilon),
        cluster_selection_method=cfg.cluster_selection_method,
        metric=cfg.metric,
        prediction_data=True,  # required for approximate_predict
    )
    clusterer.fit(X_train_final)
    train_labels = clusterer.labels_

    return preprocess, clusterer, train_plot_df, X_train, train_labels


def apply_model_to_day(
    *,
    eval_path: Path,
    cfg: HDBSCANConfig,
    preprocess: Pipeline,
    clusterer: hdbscan.HDBSCAN,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Loads eval day, samples, cleans, transforms with train preprocess, then predicts labels.
    Returns:
      eval_plot_df, X_eval, pred_labels, pred_strengths
    """
    eval_df = load_day(eval_path, cfg.features)
    eval_df_s = sample_df(eval_df, cfg.sample_fraction, cfg.sample_random_state)
    X_eval = clean_feature_frame(eval_df_s, cfg.features)

    if len(X_eval) == 0:
        raise RuntimeError(f"No usable rows in EVAL_PATH after cleaning: {eval_path}")

    eval_plot_df = eval_df_s.loc[X_eval.index]

    X_eval_final = preprocess.transform(X_eval)
    pred_labels, pred_strengths = hdbscan.approximate_predict(clusterer, X_eval_final)

    return eval_plot_df, X_eval, pred_labels, pred_strengths


def cluster_counts(labels: np.ndarray) -> tuple[int, int]:
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    return n_clusters, n_noise


def plot_clusters(
    *,
    df_plot: pd.DataFrame,
    X_index: pd.Index,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    cfg: HDBSCANConfig,
):
    """
    Scatter plot (freq vs attenuation) colored by cluster label,
    with rolling-median overlay, matching your original script behavior.
    """
    df_result = pd.DataFrame(index=X_index)
    df_result["cluster"] = labels

    unique_labels = sorted(df_result["cluster"].unique())
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    cmap = plt.colormaps.get_cmap("tab20")
    norm = mcolors.Normalize(vmin=0, vmax=max(len(unique_labels) - 1, 1))
    colors = [cmap(norm(i)) for i in range(len(unique_labels))]

    df_result["color_idx"] = df_result["cluster"].map(label_to_index)

    # Plot noise last (cluster -1 last)
    df_result_sorted = df_result.copy()
    df_result_sorted["sort_key"] = df_result_sorted["cluster"].apply(lambda x: 999 if x == -1 else x)
    df_result_sorted = df_result_sorted.sort_values("sort_key")

    n_clusters, _ = cluster_counts(labels)
    alpha = 0.25 if n_clusters <= 10 else 0.15

    plt.figure(figsize=(14, 10))

    unit = cfg.freq_unit.lower()
    if unit not in {"khz", "mhz"}:
        raise ValueError(f"Invalid freq_unit: {cfg.freq_unit}")

    freq_scale = 1000.0 if unit == "mhz" else 1.0

    x_freq = df_plot.loc[df_result_sorted.index, "carrier_frequency_kHz"] / freq_scale
    y_att = df_plot.loc[df_result_sorted.index, "attenuation_dB"]

    plt.scatter(
        x_freq,
        y_att,
        c=df_result_sorted["color_idx"],
        cmap=cmap,
        s=3,
        alpha=alpha,
        edgecolors="none",
    )

    # Rolling median overlay
    med_df = (
        df_plot.loc[df_result_sorted.index]
        .groupby("carrier_frequency_kHz")["attenuation_dB"]
        .median()
        .reset_index()
        .sort_values("carrier_frequency_kHz")
    )

    med_df["freq_plot"] = med_df["carrier_frequency_kHz"] / freq_scale

    med_df["smoothed_median"] = med_df["attenuation_dB"].rolling(window=cfg.rolling_window, center=True).median()

    plt.plot(
        med_df["freq_plot"],
        med_df["smoothed_median"],
        color="black",
        linewidth=2,
        linestyle="-",
        label="Median",
    )

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", label=str(label), markerfacecolor=colors[i], markersize=6)
        for i, label in enumerate(unique_labels)
    ]
    legend_elements.append(Line2D([0], [0], color="black", lw=2, label="Median"))

    plt.legend(
        handles=legend_elements,
        title="Cluster label",
        title_fontsize=15,
        fontsize=15,
        bbox_to_anchor=(1.025, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.xlabel(f"Sub-Carrier Frequency in {cfg.freq_unit}", fontsize=17, labelpad=10)
    plt.ylabel("Attenuation in dB", fontsize=17, labelpad=10)

    # Auto-scale based on percentiles, then fixed y-range override (as in your script)
    x = (df_plot["carrier_frequency_kHz"] / freq_scale).to_numpy()
    y = df_plot["attenuation_dB"].to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) >= 10:
        x_lo, x_hi = np.percentile(x, [0.5, 99.5])
        y_lo, y_hi = np.percentile(y, [0.5, 99.5])

        pad_x = 0.05 * (x_hi - x_lo) if x_hi > x_lo else 1.0
        pad_y = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0

        plt.xlim(x_lo - pad_x, x_hi + pad_x)
        plt.ylim(y_lo - pad_y, y_hi + pad_y)
    else:
        print("WARN: too few finite points to auto-scale")

    # Force identical y-range across plots
    plt.ylim(cfg.fixed_ylim[0], cfg.fixed_ylim[1])

    plt.title(title, fontsize=17)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()