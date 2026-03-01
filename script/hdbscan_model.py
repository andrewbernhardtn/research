"""
hdbscan_model.py
================
HDBSCAN unsupervised clustering workflow using an object-oriented design.

Module layout
-------------
HDBSCANConfig       – frozen dataclass holding all hyperparameters & settings
Data loading utils  – standalone functions: load_parquet, sample_df, clean_features, sample_tag
Preprocessor        – applies StandardScaler for feature normalisation
ClusterModel        – wraps hdbscan.HDBSCAN (fit / predict)
ClusterEvaluator    – computes internal + HDBSCAN-specific metrics per day
EvaluationReport    – aggregates per-day results → summary table + CSV export
ClusterVisualizer   – all plotting logic
HDBSCANWorkflow     – orchestrates the full train → [validate] pipeline
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import hdbscan
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Dataset = Literal["hf", "lf"]
RunMode = Literal["train_model", "validate_model"]
EvalMode = Literal["all", "dates"]

# ---------------------------------------------------------------------------
# Module-level logger  (caller can configure level / handler as needed)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HDBSCANConfig:
    """
    Immutable configuration for the HDBSCAN workflow.

    All tuneable knobs live here so that `main.py` only needs to touch
    this one object — the rest of the code is config-agnostic.
    """

    # Features used for clustering / plotting
    features: list[str]
    freq_unit: str = "kHz"          # "kHz" or "MHz"

    # HDBSCAN hyperparameters
    min_cluster_size: int = 5_000
    min_samples: int = 1_500
    cluster_selection_epsilon: float = 0.3
    cluster_selection_method: str = "eom"
    metric: str = "manhattan"

    # Sampling
    sample_fraction: Optional[float] = 0.01
    sample_random_state: int = 42

    # Plotting
    fixed_ylim: tuple[float, float] = (-80.0, 100.0)
    rolling_window: int = 7


# ---------------------------------------------------------------------------
# Default configs per dataset
# ---------------------------------------------------------------------------
_DEFAULT_CONFIGS: dict[Dataset, HDBSCANConfig] = {
    "hf": HDBSCANConfig(
        features=["attenuation_dB", "carrier_frequency_kHz"],
        freq_unit="MHz",
        min_cluster_size=5_000,
        min_samples=1_500,
        cluster_selection_epsilon=0.3,
        rolling_window=7,
        cluster_selection_method="eom",
        metric="manhattan",
    ),
    "lf": HDBSCANConfig(
        features=["attenuation_dB", "carrier_frequency_kHz"],
        freq_unit="kHz",
        min_cluster_size=8_000,
        min_samples=1_200,
        cluster_selection_epsilon=0.0,
        rolling_window=3,
        cluster_selection_method="eom",
        metric="manhattan",
    ),
}


def default_config(dataset: Dataset) -> HDBSCANConfig:
    """Return the default HDBSCANConfig for *dataset* ('hf' or 'lf')."""
    try:
        return _DEFAULT_CONFIGS[dataset]
    except KeyError:
        raise ValueError(f"Unknown dataset: {dataset!r}. Expected 'hf' or 'lf'.")


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------
def load_parquet(path: Path, cols: list[str]) -> pd.DataFrame:
    """
    Read *cols* from a parquet file and cast them to float32 in-place.

    Parameters
    ----------
    path : Path
        Location of the parquet file.
    cols : list[str]
        Column names to read (projection push-down for efficiency).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist on disk.
    """
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path, columns=cols)

    # Cast only the columns that are not already float32 (avoid copy if possible)
    needs_cast = [c for c in cols if df[c].dtype != "float32"]
    if needs_cast:
        df[needs_cast] = df[needs_cast].astype("float32", copy=False)

    return df


def sample_df(df: pd.DataFrame, frac: Optional[float], random_state: int) -> pd.DataFrame:
    """
    Return a random fraction of *df*.

    Parameters
    ----------
    frac : float or None
        Fraction to sample (0, 1].  None / >= 1.0 → return *df* unchanged.
    random_state : int
        Reproducibility seed.

    Raises
    ------
    ValueError
        If *frac* is not in (0, 1].
    """
    if frac is None or frac >= 1.0:
        return df
    if frac <= 0.0:
        raise ValueError(f"sample_fraction must be in (0, 1], got {frac}.")
    return df.sample(frac=frac, random_state=random_state)


def clean_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Select *cols* from *df*, drop rows with inf / NaN values.

    Returns a new DataFrame (original index preserved) containing only
    finite, usable rows.
    """
    X = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    return X


def sample_tag(frac: Optional[float], n: int) -> str:
    """Human-readable label describing the sample size (used in log messages)."""
    if frac is None or frac >= 1.0:
        return f"full_n={n:,}"
    return f"frac={frac:g}, n={n:,}"


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------
class Preprocessor:
    """
    Applies StandardScaler normalisation to the feature matrix.

    After calling `fit_transform`, the fitted scaler is stored and can be
    reused for transforming evaluation data via `transform`.
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the scaler on *X* and return the transformed array."""
        return self._scaler.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the already-fitted scaler to *X*."""
        return self._scaler.transform(X)

    def save(self, path: Path) -> None:
        """Persist the fitted scaler to *path* using joblib.

        Parameters
        ----------
        path : Path
            Destination file (e.g. ``save_dir / "preprocessor.joblib"``).
        """
        joblib.dump(self._scaler, path)

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        """Load a previously saved ``Preprocessor`` from *path*.

        Parameters
        ----------
        path : Path
            File produced by :meth:`save`.

        Returns
        -------
        Preprocessor
            A new instance whose internal scaler is the persisted one.
        """
        instance = cls.__new__(cls)
        instance._scaler = joblib.load(path)
        return instance


# ---------------------------------------------------------------------------
# ClusterModel
# ---------------------------------------------------------------------------
class ClusterModel:
    """
    Thin wrapper around `hdbscan.HDBSCAN` that exposes `fit` and `predict`.

    After `fit`, cluster labels are available via `self.labels_`.
    """

    def __init__(self, cfg: HDBSCANConfig) -> None:
        self.cfg = cfg
        self._clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(cfg.min_cluster_size),
            min_samples=int(cfg.min_samples),
            cluster_selection_epsilon=float(cfg.cluster_selection_epsilon),
            cluster_selection_method=cfg.cluster_selection_method,
            metric=cfg.metric,
            prediction_data=True,   # required for approximate_predict
        )
        self.labels_: Optional[np.ndarray] = None
        self.relative_validity_: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> None:
        """Fit HDBSCAN on the preprocessed feature matrix *X*."""
        self._clusterer.fit(X)
        self.labels_ = self._clusterer.labels_
        # relative_validity_ is HDBSCAN's own DBCV-based internal score
        self.relative_validity_ = getattr(self._clusterer, "relative_validity_", None)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster labels for new data using `approximate_predict`.

        Returns
        -------
        labels : np.ndarray
            Predicted cluster label per sample (-1 = noise).
        strengths : np.ndarray
            Soft membership strength per sample.
        """
        labels, strengths = hdbscan.approximate_predict(self._clusterer, X)
        return labels, strengths

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @staticmethod
    def cluster_counts(labels: np.ndarray) -> tuple[int, int]:
        """
        Return (n_clusters, n_noise) from a label array.

        Noise points are labelled -1 by HDBSCAN convention.
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        return n_clusters, n_noise

    @property
    def param_summary(self) -> str:
        """Short string of key hyperparameters for use in titles / logs."""
        c = self.cfg
        return (
            f"mc={c.min_cluster_size}, ms={c.min_samples}, "
            f"eps={c.cluster_selection_epsilon}, "
            f"method={c.cluster_selection_method}, metric={c.metric}"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Persist the fitted clusterer to *path* using joblib.

        Parameters
        ----------
        path : Path
            Destination file (e.g. ``save_dir / "cluster_model.joblib"``).
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "ClusterModel":
        """Load a previously saved ``ClusterModel`` from *path*.

        Parameters
        ----------
        path : Path
            File produced by :meth:`save`.

        Returns
        -------
        ClusterModel
            The fully fitted instance, ready to call :meth:`predict`.
        """
        return joblib.load(path)


# ---------------------------------------------------------------------------
# ClusterEvaluator
# ---------------------------------------------------------------------------
@dataclass
class DayEvalResult:
    """
    Holds all evaluation metrics for a single day's clustering result.

    Attributes
    ----------
    date_str : str
        ISO date string identifying the day (e.g. '2021-11-04').
    role : str
        Either 'train' or 'eval' — which phase produced this result.
    n_samples : int
        Number of samples that were clustered.
    n_clusters : int
        Number of clusters found (excluding noise).
    n_noise : int
        Number of points labelled as noise (-1).
    pct_assigned : float
        Percentage of points assigned to a real cluster.
    silhouette : float or None
        Silhouette score [-1, 1] — higher is better. None if < 2 clusters.
    davies_bouldin : float or None
        Davies-Bouldin index [0, ∞) — lower is better. None if < 2 clusters.
    calinski_harabasz : float or None
        Calinski-Harabasz index [0, ∞) — higher is better. None if < 2 clusters.
    relative_validity : float or None
        HDBSCAN's own DBCV-based validity score [-1, 1] — higher is better.
        Only available for the training run (approximate_predict does not
        expose this for eval days).
    """
    date_str: str
    role: str
    n_samples: int
    n_clusters: int
    n_noise: int
    pct_assigned: float
    silhouette: Optional[float]
    davies_bouldin: Optional[float]
    calinski_harabasz: Optional[float]
    relative_validity: Optional[float]

    def to_dict(self) -> dict:
        """Serialise to a plain dict (used by EvaluationReport)."""
        return {
            "date": self.date_str,
            "role": self.role,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "pct_assigned": round(self.pct_assigned, 2),
            "silhouette": self._fmt(self.silhouette),
            "davies_bouldin": self._fmt(self.davies_bouldin),
            "calinski_harabasz": self._fmt(self.calinski_harabasz),
            "relative_validity": self._fmt(self.relative_validity),
        }

    @staticmethod
    def _fmt(value: Optional[float]) -> Optional[float]:
        """Round to 4 decimal places, or keep None."""
        return round(value, 4) if value is not None else None


class ClusterEvaluator:
    """
    Computes internal clustering quality metrics for a single day.

    Keeps ``ClusterModel`` pure (fit / predict only) by owning all
    scoring logic here.

    Metrics computed
    ----------------
    - **Silhouette score** — measures how similar a point is to its own
      cluster vs. neighbouring clusters. Range [-1, 1], higher = better.
    - **Davies-Bouldin index** — ratio of within-cluster scatter to
      between-cluster separation. Range [0, ∞), lower = better.
    - **Calinski-Harabasz index** — ratio of between-cluster to
      within-cluster dispersion. Range [0, ∞), higher = better.
    - **Relative validity** — HDBSCAN's own DBCV-based score, only
      available from the training run. Range [-1, 1], higher = better.

    Notes
    -----
    All sklearn metrics are computed on **non-noise points only**. Noise
    points (label = -1) are excluded because they are by definition not
    members of any cluster and would distort the scores.

    Silhouette score uses a random subsample of at most ``max_silhouette_samples``
    points to keep runtime manageable on large datasets.
    """

    def __init__(self, max_silhouette_samples: int = 10_000) -> None:
        """
        Parameters
        ----------
        max_silhouette_samples : int
            Maximum number of (non-noise) points used for silhouette
            computation. Silhouette is O(n²) so subsampling is important
            for large datasets. Default: 10 000.
        """
        self.max_silhouette_samples = max_silhouette_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(
        self,
        *,
        X_transformed: np.ndarray,
        labels: np.ndarray,
        date_str: str,
        role: str,
        relative_validity: Optional[float] = None,
    ) -> DayEvalResult:
        """
        Compute all metrics and return a ``DayEvalResult``.

        Parameters
        ----------
        X_transformed : np.ndarray
            The preprocessed (scaled) feature matrix that was passed to
            HDBSCAN — same space the model clustered in.
        labels : np.ndarray
            HDBSCAN cluster labels (-1 = noise).
        date_str : str
            ISO date string for the day being evaluated.
        role : str
            'train' or 'eval'.
        relative_validity : float or None
            Pass ``clusterer.relative_validity_`` for training runs.
            Leave None for eval days (not available via approximate_predict).
        """
        n_samples = len(labels)
        n_clusters, n_noise = ClusterModel.cluster_counts(labels)
        pct_assigned = 100.0 * (1.0 - n_noise / max(n_samples, 1))

        silhouette, davies_bouldin, calinski_harabasz = self._score_metrics(
            X_transformed, labels, n_clusters
        )

        result = DayEvalResult(
            date_str=date_str,
            role=role,
            n_samples=n_samples,
            n_clusters=n_clusters,
            n_noise=n_noise,
            pct_assigned=pct_assigned,
            silhouette=silhouette,
            davies_bouldin=davies_bouldin,
            calinski_harabasz=calinski_harabasz,
            relative_validity=relative_validity,
        )

        self._log_result(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _score_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute the three sklearn internal metrics.

        Returns (silhouette, davies_bouldin, calinski_harabasz).
        All three return None when fewer than 2 clusters are present,
        since the metrics are undefined in that case.
        """
        if n_clusters < 2:
            logger.warning(
                "EVAL  | only %d cluster(s) found — sklearn metrics require ≥ 2. "
                "Skipping silhouette / DB / CH.",
                n_clusters,
            )
            return None, None, None

        # Exclude noise points (-1) from all metric calculations
        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]

        if len(np.unique(labels_clean)) < 2:
            return None, None, None

        # Cast to float64 — sklearn's distance matrix internals can overflow
        # with float32 on large datasets, producing divide-by-zero warnings.
        X_clean64 = X_clean.astype(np.float64, copy=False)

        # Silhouette: subsample to bound O(n²) cost
        X_sil, labels_sil = self._subsample_for_silhouette(X_clean64, labels_clean)

        # Suppress residual numerical warnings that are non-fatal and expected
        # at this scale — the scores are still valid after float64 cast.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            silhouette = float(silhouette_score(X_sil, labels_sil))
            davies_bouldin = float(davies_bouldin_score(X_clean64, labels_clean))
            calinski_harabasz = float(calinski_harabasz_score(X_clean64, labels_clean))

        return silhouette, davies_bouldin, calinski_harabasz

    def _subsample_for_silhouette(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly subsample *X* and *labels* to at most
        ``self.max_silhouette_samples`` rows for silhouette computation.
        """
        n = len(X)
        if n <= self.max_silhouette_samples:
            return X, labels
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(n, size=self.max_silhouette_samples, replace=False)
        return X[idx], labels[idx]

    @staticmethod
    def _log_result(result: DayEvalResult) -> None:
        """Emit a structured log line for this day's metrics."""
        logger.info(
            "METRICS | %s (%s) | clusters=%d, noise=%d, assigned=%.1f%% | "
            "silhouette=%s, DB=%s, CH=%s, rel_validity=%s",
            result.date_str,
            result.role,
            result.n_clusters,
            result.n_noise,
            result.pct_assigned,
            _fmt_metric(result.silhouette),
            _fmt_metric(result.davies_bouldin),
            _fmt_metric(result.calinski_harabasz),
            _fmt_metric(result.relative_validity),
        )


def _fmt_metric(value: Optional[float]) -> str:
    """Format a metric value for log output ('N/A' if None)."""
    return f"{value:.4f}" if value is not None else "N/A"


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------
class EvaluationReport:
    """
    Accumulates ``DayEvalResult`` objects across all days and produces a
    summary table (printed to log) and an optional CSV export.

    Usage
    -----
    ::

        report = EvaluationReport()
        report.add(train_result)          # add training day result
        for result in eval_results:
            report.add(result)            # add each eval day result
        report.save_csv(save_dir / "evaluation_summary.csv")
        report.log_summary()

    The CSV has one row per day with all metrics as columns, making it
    easy to import into Excel or a Jupyter notebook for further analysis.
    """

    def __init__(self) -> None:
        self._results: list[DayEvalResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, result: DayEvalResult) -> None:
        """Append a single day's result to the report."""
        self._results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return all accumulated results as a tidy ``pd.DataFrame``.

        Rows are ordered: training day first, then eval days by date.
        """
        if not self._results:
            return pd.DataFrame()
        rows = [r.to_dict() for r in self._results]
        df = pd.DataFrame(rows)
        # Training row first, then eval rows sorted by date
        train_rows = df[df["role"] == "train"]
        eval_rows = df[df["role"] == "eval"].sort_values("date")
        return pd.concat([train_rows, eval_rows], ignore_index=True)

    def save_csv(self, out_path: Path) -> None:
        """
        Write the summary table to *out_path* as a CSV file.

        Parameters
        ----------
        out_path : Path
            Destination path (e.g. ``results/run_01/evaluation_summary.csv``).
        """
        df = self.to_dataframe()
        if df.empty:
            logger.warning("REPORT | No results to save — skipping CSV export.")
            return
        df.to_csv(out_path, index=False)
        logger.info("REPORT | Saved evaluation summary → %s", out_path.name)

    def log_summary(self) -> None:
        """Print a human-readable summary table to the logger."""
        df = self.to_dataframe()
        if df.empty:
            logger.info("REPORT | No results to summarise.")
            return

        logger.info("REPORT | Evaluation summary (%d day(s)):", len(df))
        logger.info(
            "%-14s %-6s %8s %9s %7s %10s %11s %9s %13s %16s",
            "date", "role", "samples", "clusters", "noise",
            "assigned%", "silhouette", "DB_index", "CH_index", "rel_validity",
        )
        logger.info("-" * 100)
        for _, row in df.iterrows():
            logger.info(
                "%-14s %-6s %8d %9d %7d %9.1f%% %11s %9s %13s %16s",
                row["date"],
                row["role"],
                row["n_samples"],
                row["n_clusters"],
                row["n_noise"],
                row["pct_assigned"],
                _fmt_metric(row["silhouette"]),
                _fmt_metric(row["davies_bouldin"]),
                _fmt_metric(row["calinski_harabasz"]),
                _fmt_metric(row["relative_validity"]),
            )


# ---------------------------------------------------------------------------
# ClusterVisualizer
# ---------------------------------------------------------------------------
class ClusterVisualizer:
    """
    All plotting logic for HDBSCAN results.

    The single public method `plot` produces a scatter plot of
    (frequency, attenuation) coloured by cluster label, overlaid with a
    smoothed median line, and saves the figure to disk.
    """

    # Consistent figure size across all plots
    _FIG_SIZE: tuple[int, int] = (14, 10)
    _DPI: int = 200

    def __init__(self, cfg: HDBSCANConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot(
        self,
        *,
        df_plot: pd.DataFrame,
        X_index: pd.Index,
        labels: np.ndarray,
        title: str,
        out_path: Path,
    ) -> None:
        """
        Save a cluster scatter plot to *out_path*.

        Parameters
        ----------
        df_plot : pd.DataFrame
            Raw (unscaled) data for the day — used for axis values.
        X_index : pd.Index
            Index of rows that were actually clustered (after cleaning/sampling).
        labels : np.ndarray
            HDBSCAN cluster labels aligned with *X_index*.
        title : str
            Plot title.
        out_path : Path
            Destination file (PNG).
        """
        df_result, colors, cmap, freq_scale = self._prepare_plot_data(df_plot, X_index, labels)

        fig, ax = plt.subplots(figsize=self._FIG_SIZE)

        self._draw_scatter(ax, df_plot, df_result, freq_scale, colors, cmap, labels)
        self._draw_median_line(ax, df_plot, df_result, freq_scale)
        self._set_axis_limits(ax, df_plot, freq_scale)
        self._add_legend(ax, df_result, colors)
        self._apply_labels_and_title(ax, title)

        fig.tight_layout()
        fig.savefig(out_path, dpi=self._DPI)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Private helpers  (each handles one visual concern)
    # ------------------------------------------------------------------
    def _freq_scale(self) -> float:
        """Return the divisor to convert kHz → display unit."""
        unit = self.cfg.freq_unit.lower()
        if unit == "mhz":
            return 1_000.0
        if unit == "khz":
            return 1.0
        raise ValueError(f"Invalid freq_unit: {self.cfg.freq_unit!r}. Expected 'kHz' or 'MHz'.")

    def _prepare_plot_data(
        self,
        df_plot: pd.DataFrame,
        X_index: pd.Index,
        labels: np.ndarray,
    ) -> tuple[pd.DataFrame, list, plt.cm.ScalarMappable, float]:
        """
        Build the result DataFrame and colour map needed for plotting.

        Returns
        -------
        df_result : pd.DataFrame
            Contains 'cluster', 'color_idx', 'sort_key' columns, sorted so
            that noise points (-1) are drawn first (under real clusters).
        colors : list
            RGBA colour per unique label (index matches label_to_index).
        freq_scale : float
            Divisor to convert carrier_frequency_kHz to the display unit.
        """
        df_result = pd.DataFrame({"cluster": labels}, index=X_index)

        unique_labels = sorted(df_result["cluster"].unique())
        label_to_index = {lbl: i for i, lbl in enumerate(unique_labels)}

        cmap = plt.colormaps.get_cmap("tab20")
        norm = mcolors.Normalize(vmin=0, vmax=max(len(unique_labels) - 1, 1))
        colors = [cmap(norm(i)) for i in range(len(unique_labels))]

        df_result["color_idx"] = df_result["cluster"].map(label_to_index)
        # Sort so noise (-1) is rendered first (drawn underneath clusters)
        # df_result["sort_key"] = df_result["cluster"].apply(lambda x: 999 if x == -1 else x)
        df_result["sort_key"] = df_result["cluster"].where(df_result["cluster"] != -1, other=df_result["cluster"].max() + 1)
        df_result.sort_values("sort_key", inplace=True)

        return df_result, colors, cmap, self._freq_scale()

    def _draw_scatter(
        self,
        ax: plt.Axes,
        df_plot: pd.DataFrame,
        df_result: pd.DataFrame,
        freq_scale: float,
        colors: list,
        cmap: mcolors.Colormap,
        labels: np.ndarray,
    ) -> None:
        """Draw the main cluster scatter on *ax*."""
        n_clusters, _ = ClusterModel.cluster_counts(labels)
        alpha = 0.25 if n_clusters <= 10 else 0.15

        x = df_plot.loc[df_result.index, "carrier_frequency_kHz"] / freq_scale
        y = df_plot.loc[df_result.index, "attenuation_dB"]

        ax.scatter(
            x, y,
            c=df_result["color_idx"],
            cmap=cmap,
            s=3,
            alpha=alpha,
            edgecolors="none",
        )

    def _draw_median_line(
        self,
        ax: plt.Axes,
        df_plot: pd.DataFrame,
        df_result: pd.DataFrame,
        freq_scale: float,
    ) -> None:
        """Overlay a smoothed per-frequency median attenuation line on *ax*."""
        med_df = (
            df_plot.loc[df_result.index]
            .groupby("carrier_frequency_kHz")["attenuation_dB"]
            .median()
            .reset_index()
            .sort_values("carrier_frequency_kHz")
        )
        med_df["freq_plot"] = med_df["carrier_frequency_kHz"] / freq_scale
        med_df["smoothed_median"] = (
            med_df["attenuation_dB"]
            .rolling(window=self.cfg.rolling_window, center=True)
            .median()
        )
        ax.plot(
            med_df["freq_plot"],
            med_df["smoothed_median"],
            color="black",
            linewidth=2,
            linestyle="-",
            label="Median",
        )

    def _set_axis_limits(
        self,
        ax: plt.Axes,
        df_plot: pd.DataFrame,
        freq_scale: float,
    ) -> None:
        """
        Auto-scale x-axis to the 0.5–99.5 percentile range of the raw data,
        then apply the fixed y-axis limits from config.
        """
        x = (df_plot["carrier_frequency_kHz"] / freq_scale).to_numpy()
        y = df_plot["attenuation_dB"].to_numpy()

        finite_mask = np.isfinite(x) & np.isfinite(y)
        x_finite, y_finite = x[finite_mask], y[finite_mask]

        if len(x_finite) >= 10:
            x_lo, x_hi = np.percentile(x_finite, [0.5, 99.5])
            pad_x = 0.05 * (x_hi - x_lo) if x_hi > x_lo else 1.0
            ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
        else:
            logger.warning("Too few finite points to auto-scale x-axis.")

        ax.set_ylim(*self.cfg.fixed_ylim)

    def _add_legend(
        self,
        ax: plt.Axes,
        df_result: pd.DataFrame,
        colors: list,
    ) -> None:
        """Build and attach the cluster-label legend to *ax*."""
        unique_labels = sorted(df_result["cluster"].unique())
        legend_elements = [
            Line2D(
                [0], [0],
                marker="o", color="none",
                label=str(lbl),
                markerfacecolor=colors[i],
                markersize=6,
            )
            for i, lbl in enumerate(unique_labels)
        ]
        legend_elements.append(Line2D([0], [0], color="black", lw=2, label="Median"))

        ax.legend(
            handles=legend_elements,
            title="Cluster label",
            title_fontsize=15,
            fontsize=15,
            bbox_to_anchor=(1.025, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

    def _apply_labels_and_title(self, ax: plt.Axes, title: str) -> None:
        """Set axis labels and the plot title."""
        ax.set_xlabel(f"Sub-Carrier Frequency in {self.cfg.freq_unit}", fontsize=17, labelpad=10)
        ax.set_ylabel("Attenuation in dB", fontsize=17, labelpad=10)
        ax.set_title(title, fontsize=17)


# ---------------------------------------------------------------------------
# HDBSCANWorkflow  (orchestrator)
# ---------------------------------------------------------------------------
class HDBSCANWorkflow:
    """
    Top-level orchestrator for the HDBSCAN train → [validate] pipeline.

    This class owns *only* workflow logic: path resolution, loop control,
    logging, and delegating to the focused sub-classes above.

    Parameters
    ----------
    train_path : Path
        Parquet file used to train the reference HDBSCAN model.
    data_dir : Path
        Directory that contains all parquet files for the dataset.
    results_dir : Path
        Root directory where output folders are created.
    output_folder : str
        Name of the output subfolder inside *results_dir*.
    run_mode : RunMode
        ``'train_model'`` → train and plot only.
        ``'validate_model'`` → train, then predict on eval files.
    eval_mode : EvalMode
        ``'all'`` → use every file in *data_dir* (except train).
        ``'dates'`` → use only files whose date matches *eval_dates*.
    eval_dates : set[str] or None
        ISO date strings (``'YYYY-MM-DD'``) to evaluate. Only used
        when ``eval_mode='dates'``.
    cfg : HDBSCANConfig or None
        Supply a custom config, or leave ``None`` to use the dataset
        default (inferred from the train filename prefix).

    Usage
    -----
    ::

        wf = HDBSCANWorkflow(
            train_path=Path("data/lf_2021-11-25.parquet"),
            data_dir=Path("data"),
            results_dir=Path("results"),
            output_folder="experiment_01",
            run_mode="validate_model",
            eval_mode="dates",
            eval_dates={"2021-11-04"},
        )
        wf.run()
    """

    def __init__(
        self,
        *,
        train_path: Path,
        data_dir: Path,
        results_dir: Path,
        output_folder: str,
        run_mode: RunMode = "validate_model",
        eval_mode: EvalMode = "all",
        eval_dates: Optional[set[str]] = None,
        cfg: Optional[HDBSCANConfig] = None,
    ) -> None:
        self.train_path = train_path
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.output_folder = output_folder
        self.run_mode = run_mode
        self.eval_mode = eval_mode
        self.eval_dates: set[str] = eval_dates or set()

        self.dataset: Dataset = self._infer_dataset(train_path)
        self.cfg: HDBSCANConfig = cfg or default_config(self.dataset)

        self.save_dir: Path = self._ensure_dir(results_dir / output_folder)
        self.train_date: str = self._date_from_stem(train_path)
        self.eval_paths: list[Path] = self._select_eval_paths()

        # Sub-components
        self._visualizer = ClusterVisualizer(self.cfg)
        self._evaluator = ClusterEvaluator()
        self._report = EvaluationReport()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute the full workflow:

        1. Saves the run configuration to ``config.json`` for reproducibility.
        2. Always trains the HDBSCAN model on ``train_path``.
        3. If ``run_mode='validate_model'``, applies the fitted model to
           every path in ``eval_paths`` and saves a plot for each.
        4. Saves an evaluation summary CSV and logs a metrics table.
        """
        self._validate_run_mode()
        self._save_config()

        # -- Training phase --
        preprocessor, model, train_plot_df, X_train, X_train_transformed = self._train(self.train_path)
        self._log_train_result(model, X_train)
        self._save_plot(
            model=model,
            plot_df=train_plot_df,
            X_index=X_train.index,
            labels=model.labels_,
            date_str=self.train_date,
            stem=self.train_path.stem,
        )

        # Evaluate training day and add to report
        train_result = self._evaluator.evaluate(
            X_transformed=X_train_transformed,
            labels=model.labels_,
            date_str=self.train_date,
            role="train",
            relative_validity=model.relative_validity_,
        )
        self._report.add(train_result)
        self._save_model(preprocessor, model)

        if self.run_mode == "train_model":
            self._finalise_report()
            return

        # -- Validation phase --
        for eval_path in self.eval_paths:
            self._run_single_eval(eval_path, preprocessor, model)

        self._finalise_report()

    # ------------------------------------------------------------------
    # Config saving (reproducibility)
    # ------------------------------------------------------------------
    def _save_config(self) -> None:
        """
        Write the full run configuration to ``config.json`` inside the
        results folder so that any experiment can be reproduced later.

        The JSON file captures both the workflow settings (paths, run mode,
        eval dates) and every hyperparameter from ``HDBSCANConfig``.
        """
        config_record = {
            "train_path": str(self.train_path),
            "dataset": self.dataset,
            "output_folder": self.output_folder,
            "run_mode": self.run_mode,
            "eval_mode": self.eval_mode,
            "eval_dates": sorted(self.eval_dates),
            **asdict(self.cfg),
        }

        # Convert tuple fields to lists for clean JSON serialisation
        for key, value in config_record.items():
            if isinstance(value, tuple):
                config_record[key] = list(value)

        out_path = self.save_dir / "config.json"
        out_path.write_text(json.dumps(config_record, indent=4))
        logger.info("CONFIG | saved → %s", out_path.name)

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------
    def _save_model(
        self,
        preprocessor: Preprocessor,
        model: ClusterModel,
    ) -> None:
        """Save the fitted preprocessor and cluster model to ``self.save_dir``.

        Writes two files alongside ``config.json``:

        - ``preprocessor.joblib`` — the fitted :class:`Preprocessor`
        - ``cluster_model.joblib`` — the fitted :class:`ClusterModel`

        Load these with::

            preprocessor = Preprocessor.load(save_dir / "preprocessor.joblib")
            model = ClusterModel.load(save_dir / "cluster_model.joblib")

        Parameters
        ----------
        preprocessor : Preprocessor
            The fitted preprocessor returned by :meth:`_train`.
        model : ClusterModel
            The fitted cluster model returned by :meth:`_train`.
        """
        preprocessor_path = self.save_dir / "preprocessor.joblib"
        model_path = self.save_dir / "cluster_model.joblib"

        preprocessor.save(preprocessor_path)
        logger.info("MODEL | preprocessor saved → %s", preprocessor_path.name)

        model.save(model_path)
        logger.info("MODEL | cluster model saved → %s", model_path.name)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def _train(
        self, path: Path
    ) -> tuple[Preprocessor, ClusterModel, pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Load, preprocess, and fit HDBSCAN on *path*.

        Returns
        -------
        preprocessor, model, plot_df, X, X_transformed
            X_transformed is returned so the evaluator can score metrics
            in the same feature space the model was trained in.
        """
        df = load_parquet(path, self.cfg.features)
        df_sampled = sample_df(df, self.cfg.sample_fraction, self.cfg.sample_random_state)
        X = clean_features(df_sampled, self.cfg.features)

        if X.empty:
            raise RuntimeError(f"No usable rows after cleaning: {path}")

        plot_df = df_sampled.loc[X.index]

        preprocessor = Preprocessor()
        X_transformed = preprocessor.fit_transform(X)

        model = ClusterModel(self.cfg)
        model.fit(X_transformed)

        return preprocessor, model, plot_df, X, X_transformed

    def _log_train_result(self, model: ClusterModel, X_train: pd.DataFrame) -> None:
        """Emit a structured log line summarising the training outcome."""
        n_clusters, n_noise = ClusterModel.cluster_counts(model.labels_)
        tag = sample_tag(self.cfg.sample_fraction, len(X_train))
        logger.info(
            "TRAIN | file=%s | %s | clusters=%d, noise=%d | %s",
            self.train_path.stem, tag, n_clusters, n_noise,
            model.param_summary,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _run_single_eval(
        self,
        eval_path: Path,
        preprocessor: Preprocessor,
        model: ClusterModel,
    ) -> None:
        """Apply the fitted model to *eval_path*, evaluate, and save a plot."""
        try:
            plot_df, X_eval, X_eval_transformed, labels = self._predict(
                eval_path, preprocessor, model
            )
        except RuntimeError as exc:
            logger.warning("SKIP | file=%s | reason: %s", eval_path.stem, exc)
            return

        n_clusters, n_noise = ClusterModel.cluster_counts(labels)
        pct_assigned = 100.0 * (1.0 - n_noise / max(len(labels), 1))
        tag = sample_tag(self.cfg.sample_fraction, len(X_eval))
        logger.info(
            "EVAL  | file=%s | %s | pred_clusters=%d, noise=%d (%.1f%% assigned)",
            eval_path.stem, tag, n_clusters, n_noise, pct_assigned,
        )

        # Compute and record evaluation metrics for this day
        eval_date = self._date_from_stem(eval_path)
        eval_result = self._evaluator.evaluate(
            X_transformed=X_eval_transformed,
            labels=labels,
            date_str=eval_date,
            role="eval",
            relative_validity=None,  # not available via approximate_predict
        )
        self._report.add(eval_result)

        self._save_plot(
            model=model,
            plot_df=plot_df,
            X_index=X_eval.index,
            labels=labels,
            date_str=eval_date,
            stem=eval_path.stem,
        )

    def _predict(
        self,
        eval_path: Path,
        preprocessor: Preprocessor,
        model: ClusterModel,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Load *eval_path*, transform it, and return predicted labels.

        Returns
        -------
        plot_df, X_eval, X_eval_transformed, labels
            X_eval_transformed is returned so the evaluator can score
            metrics in the same feature space as the training data.
        """
        df = load_parquet(eval_path, self.cfg.features)
        df_sampled = sample_df(df, self.cfg.sample_fraction, self.cfg.sample_random_state)
        X = clean_features(df_sampled, self.cfg.features)

        if X.empty:
            raise RuntimeError(f"No usable rows after cleaning: {eval_path}")

        plot_df = df_sampled.loc[X.index]
        X_transformed = preprocessor.transform(X)
        labels, _strengths = model.predict(X_transformed)
        return plot_df, X, X_transformed, labels

    # ------------------------------------------------------------------
    # Report helper
    # ------------------------------------------------------------------
    def _finalise_report(self) -> None:
        """Log the metrics summary table and save the CSV report."""
        self._report.log_summary()
        csv_path = self.save_dir / "evaluation_summary.csv"
        self._report.save_csv(csv_path)

    # ------------------------------------------------------------------
    # Plot helper
    # ------------------------------------------------------------------
    def _save_plot(
        self,
        *,
        model: ClusterModel,
        plot_df: pd.DataFrame,
        X_index: pd.Index,
        labels: np.ndarray,
        date_str: str,
        stem: str,
    ) -> None:
        """Build plot title, resolve output path, delegate to visualizer."""
        title = f"{stem} | {model.param_summary}"
        out_path = self._build_plot_path(date_str)

        self._visualizer.plot(
            df_plot=plot_df,
            X_index=X_index,
            labels=labels,
            title=title,
            out_path=out_path,
        )
        logger.info("PLOT  | saved → %s", out_path.name)

    # ------------------------------------------------------------------
    # Path / filesystem helpers
    # ------------------------------------------------------------------
    def _build_plot_path(self, date_str: str) -> Path:
        """Construct the output PNG filename from config parameters."""
        c = self.cfg
        filename = (
            f"hdbscan_{self.dataset}_{date_str}_"
            f"mc{c.min_cluster_size}_"
            f"ms{c.min_samples}_"
            f"eps{int(c.cluster_selection_epsilon * 1000)}_"
            f"{c.cluster_selection_method}_"
            f"{c.metric}.png"
        )
        return self.save_dir / filename

    def _select_eval_paths(self) -> list[Path]:
        """Discover and filter evaluation parquet files."""
        pattern = f"{self.dataset}_*.parquet"
        all_paths = sorted(self.data_dir.glob(pattern))
        candidates = [p for p in all_paths if p.resolve() != self.train_path.resolve()]
        return self._filter_by_mode(candidates)

    def _filter_by_mode(self, paths: list[Path]) -> list[Path]:
        """Apply eval_mode / eval_dates filter to *paths*."""
        if self.eval_mode == "all":
            return paths
        if self.eval_mode == "dates":
            return [p for p in paths if self._date_from_stem(p) in self.eval_dates]
        raise ValueError(f"eval_mode must be 'all' or 'dates', got {self.eval_mode!r}.")

    def _validate_run_mode(self) -> None:
        """Raise early if run_mode + eval_paths combination is invalid."""
        if self.run_mode not in {"train_model", "validate_model"}:
            raise ValueError(
                f"run_mode must be 'train_model' or 'validate_model', got {self.run_mode!r}."
            )
        if self.run_mode == "validate_model" and not self.eval_paths:
            raise ValueError(
                "run_mode='validate_model' but no evaluation files were selected.\n"
                "  • eval_mode='dates': check eval_dates matches available parquet dates.\n"
                "  • eval_mode='all': ensure there are other parquet files besides train_path."
            )

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_dataset(path: Path) -> Dataset:
        """Infer 'hf' or 'lf' from the filename prefix."""
        name = path.name.lower()
        if name.startswith("hf_"):
            return "hf"
        if name.startswith("lf_"):
            return "lf"
        raise ValueError(
            f"Cannot infer dataset type from filename: {path.name!r}. "
            "Expected prefix 'hf_' or 'lf_'."
        )

    @staticmethod
    def _date_from_stem(path: Path) -> str:
        """Extract the date portion from a stem like 'hf_2021-11-13'."""
        parts = path.stem.split("_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Expected filename like hf_YYYY-MM-DD.parquet, got: {path.name!r}."
            )
        return parts[1]

    @staticmethod
    def _ensure_dir(path: Path) -> Path:
        """Create *path* (and parents) if it does not exist; return *path*."""
        path.mkdir(parents=True, exist_ok=True)
        return path
