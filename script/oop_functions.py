from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

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
RunMode = Literal["train_model", "validate_model"]
EvalMode = Literal["all", "dates"]


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


class HDBSCANWorkflow:
    """
    Single entry-point class for your HDBSCAN training + validation workflow.

    Usage (from run_hdbscan.py):
        wf = HDBSCANWorkflow(
            train_path=...,
            data_dir=...,
            results_dir=...,
            run_name="...",
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
        run_name: str,
        run_mode: RunMode = "validate_model",
        eval_mode: EvalMode = "all",
        eval_dates: Optional[set[str]] = None,
        cfg: Optional[HDBSCANConfig] = None,
    ) -> None:
        self.train_path = train_path
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.run_name = run_name
        self.run_mode = run_mode
        self.eval_mode = eval_mode
        self.eval_dates = eval_dates or set()

        self.dataset: Dataset = self.infer_dataset_from_filename(train_path)
        self.cfg: HDBSCANConfig = cfg or self.default_hdbscan_cfg(self.dataset)

        self.save_dir = self.ensure_dir(self.results_dir / self.run_name)

        self.train_date = self.date_from_stem(self.train_path)
        self.eval_paths = self._select_eval_paths()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(self) -> None:
        """
        Runs training always, then optionally validation (depending on run_mode).
        Produces plots into results_dir/run_name.
        """
        if self.run_mode == "validate_model" and not self.eval_paths:
            raise ValueError(
                "RUN_MODE='validate_model' but no evaluation files were selected.\n"
                "• If EVAL_MODE='dates': check EVAL_DATES matches available parquet dates.\n"
                "• If EVAL_MODE='all': ensure there are other parquet files besides TRAIN_PATH."
            )

        # 1) Train model
        preprocess, clusterer, train_plot_df, X_train, train_labels = self.fit_reference_model()

        n_train_clusters, n_train_noise = self.cluster_counts(train_labels)
        train_sample_tag = self.sample_tag(self.cfg.sample_fraction, len(X_train))
        pca_str = self.pca_info(self.cfg)

        print(
            f"✅ TRAIN_MODEL fitted: {self.train_path.stem} | {train_sample_tag} | "
            f"Clusters={n_train_clusters}, Noise={n_train_noise} | "
            f"mc={self.cfg.min_cluster_size}, ms={self.cfg.min_samples}, eps={self.cfg.cluster_selection_epsilon}, "
            f"method={self.cfg.cluster_selection_method}, metric={self.cfg.metric} {pca_str}"
        )

        train_title = (
            f"{self.train_path.stem} | "
            f"mc={self.cfg.min_cluster_size}, ms={self.cfg.min_samples}, eps={self.cfg.cluster_selection_epsilon} | "
            f"method={self.cfg.cluster_selection_method}, metric={self.cfg.metric}"
        )

        train_out = self.build_plot_out_path(self.save_dir, dataset=self.dataset, date_str=self.train_date, cfg=self.cfg)
        self.plot_clusters(
            df_plot=train_plot_df,
            X_index=X_train.index,
            labels=train_labels,
            title=train_title,
            out_path=train_out,
            cfg=self.cfg,
        )
        print(f"   ↳ saved train plot: {train_out.name}")

        # 2) Validate model (optional)
        if self.run_mode == "train_model":
            return

        if self.run_mode != "validate_model":
            raise ValueError(f"Unknown RUN_MODE: {self.run_mode}")

        for eval_path in self.eval_paths:
            try:
                eval_plot_df, X_eval, pred_labels, _pred_strengths = self.apply_model_to_day(
                    eval_path=eval_path,
                    cfg=self.cfg,
                    preprocess=preprocess,
                    clusterer=clusterer,
                )
            except RuntimeError as e:
                print(f"⚠️  {eval_path.stem}: {e} | skipping.")
                continue

            n_eval_clusters, n_eval_noise = self.cluster_counts(pred_labels)
            pct_assigned = 100.0 * (1.0 - (n_eval_noise / max(len(pred_labels), 1)))
            eval_sample_tag = self.sample_tag(self.cfg.sample_fraction, len(X_eval))

            print(
                f"✅ VALIDATE_MODEL applied: {eval_path.stem} | {eval_sample_tag} | "
                f"PredClusters={n_eval_clusters}, PredNoise={n_eval_noise} ({pct_assigned:.1f}% assigned)"
            )

            eval_title = (
                f"{eval_path.stem} | "
                f"mc={self.cfg.min_cluster_size}, ms={self.cfg.min_samples}, eps={self.cfg.cluster_selection_epsilon} | "
                f"method={self.cfg.cluster_selection_method}, metric={self.cfg.metric}"
            )

            eval_date = self.date_from_stem(eval_path)
            eval_out = self.build_plot_out_path(self.save_dir, dataset=self.dataset, date_str=eval_date, cfg=self.cfg)

            self.plot_clusters(
                df_plot=eval_plot_df,
                X_index=X_eval.index,
                labels=pred_labels,
                title=eval_title,
                out_path=eval_out,
                cfg=self.cfg,
            )
            print(f"   ↳ saved eval plot: {eval_out.name}")

    # ---------------------------------------------------------------------
    # Path / config helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def infer_dataset_from_filename(path: Path) -> Dataset:
        name = path.name.lower()
        if name.startswith("hf_"):
            return "hf"
        if name.startswith("lf_"):
            return "lf"
        raise ValueError(f"Cannot infer dataset type from filename: {path.name}. Expected prefix hf_ or lf_")

    @staticmethod
    def date_from_stem(path: Path) -> str:
        parts = path.stem.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Expected filename like hf_YYYY-MM-DD.parquet or lf_YYYY-MM-DD.parquet, got: {path.name}")
        return parts[1]

    @staticmethod
    def discover_eval_paths(data_dir: Path, dataset: Dataset, exclude: Path) -> list[Path]:
        pattern = f"{dataset}_*.parquet"
        files = sorted(data_dir.glob(pattern))
        return [p for p in files if p.resolve() != exclude.resolve()]

    @staticmethod
    def filter_eval_paths(paths: list[Path], mode: EvalMode, dates: set[str]) -> list[Path]:
        paths = sorted(paths, key=lambda p: p.name)
        if mode == "all":
            return paths
        if mode == "dates":
            return [p for p in paths if HDBSCANWorkflow.date_from_stem(p) in dates]
        raise ValueError("EVAL_MODE must be one of: all | dates")

    @staticmethod
    def default_hdbscan_cfg(dataset: Dataset) -> HDBSCANConfig:
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

    @staticmethod
    def build_plot_out_path(save_dir: Path, *, dataset: Dataset, date_str: str, cfg: HDBSCANConfig) -> Path:
        return save_dir / (
            f"hdbscan_{dataset}_{date_str}_"
            f"mc{cfg.min_cluster_size}_"
            f"ms{cfg.min_samples}_"
            f"eps{int(cfg.cluster_selection_epsilon * 1000)}_"
            f"{cfg.cluster_selection_method}_"
            f"{cfg.metric}.png"
        )

    @staticmethod
    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _select_eval_paths(self) -> list[Path]:
        all_eval_paths = self.discover_eval_paths(self.data_dir, self.dataset, exclude=self.train_path)
        return self.filter_eval_paths(all_eval_paths, mode=self.eval_mode, dates=set(self.eval_dates))

    # ---------------------------------------------------------------------
    # Core ML pieces
    # ---------------------------------------------------------------------
    @staticmethod
    def build_preprocess_pipeline(cfg: HDBSCANConfig) -> Pipeline:
        steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
        if cfg.use_pca:
            steps.append(("pca", PCA(n_components=cfg.pca_n_components, svd_solver="randomized", random_state=42)))
        return Pipeline(steps)

    @staticmethod
    def sample_df(df: pd.DataFrame, frac: Optional[float], random_state: int) -> pd.DataFrame:
        if frac is None or frac >= 1:
            return df
        if frac <= 0:
            raise ValueError("sample_fraction must be > 0, or use None/1.0 for no sampling.")
        return df.sample(frac=frac, random_state=random_state)

    @staticmethod
    def load_day(path: Path, cols: list[str]) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet: {path}")
        df = pd.read_parquet(path, columns=cols)
        if any(df[c].dtype != "float32" for c in cols):
            df[cols] = df[cols].astype("float32", copy=False)
        return df

    @staticmethod
    def clean_feature_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        X = df[cols]
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        return X

    @staticmethod
    def sample_tag(frac: Optional[float], n: int) -> str:
        if frac is None or frac >= 1:
            return f"full_n{n}"
        return f"s{frac:g}_n{n}"

    @staticmethod
    def pca_info(cfg: HDBSCANConfig) -> str:
        return f"PCA({cfg.pca_n_components}D)" if cfg.use_pca else ""

    @staticmethod
    def cluster_counts(labels: np.ndarray) -> tuple[int, int]:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        return n_clusters, n_noise

    def fit_reference_model(
        self,
    ) -> tuple[Pipeline, hdbscan.HDBSCAN, pd.DataFrame, pd.DataFrame, np.ndarray]:
        train_df = self.load_day(self.train_path, self.cfg.features)
        train_df_s = self.sample_df(train_df, self.cfg.sample_fraction, self.cfg.sample_random_state)
        X_train = self.clean_feature_frame(train_df_s, self.cfg.features)

        if len(X_train) == 0:
            raise RuntimeError(f"No usable rows in TRAIN_PATH after cleaning: {self.train_path}")

        train_plot_df = train_df_s.loc[X_train.index]

        preprocess = self.build_preprocess_pipeline(self.cfg)
        preprocess.fit(X_train)
        X_train_final = preprocess.transform(X_train)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(self.cfg.min_cluster_size),
            min_samples=int(self.cfg.min_samples),
            cluster_selection_epsilon=float(self.cfg.cluster_selection_epsilon),
            cluster_selection_method=self.cfg.cluster_selection_method,
            metric=self.cfg.metric,
            prediction_data=True,
        )
        clusterer.fit(X_train_final)
        train_labels = clusterer.labels_
        return preprocess, clusterer, train_plot_df, X_train, train_labels

    @staticmethod
    def apply_model_to_day(
        *,
        eval_path: Path,
        cfg: HDBSCANConfig,
        preprocess: Pipeline,
        clusterer: hdbscan.HDBSCAN,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        eval_df = HDBSCANWorkflow.load_day(eval_path, cfg.features)
        eval_df_s = HDBSCANWorkflow.sample_df(eval_df, cfg.sample_fraction, cfg.sample_random_state)
        X_eval = HDBSCANWorkflow.clean_feature_frame(eval_df_s, cfg.features)

        if len(X_eval) == 0:
            raise RuntimeError(f"No usable rows in EVAL_PATH after cleaning: {eval_path}")

        eval_plot_df = eval_df_s.loc[X_eval.index]
        X_eval_final = preprocess.transform(X_eval)
        pred_labels, pred_strengths = hdbscan.approximate_predict(clusterer, X_eval_final)
        return eval_plot_df, X_eval, pred_labels, pred_strengths

    # ---------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------
    @staticmethod
    def plot_clusters(
        *,
        df_plot: pd.DataFrame,
        X_index: pd.Index,
        labels: np.ndarray,
        title: str,
        out_path: Path,
        cfg: HDBSCANConfig,
    ) -> None:
        df_result = pd.DataFrame(index=X_index)
        df_result["cluster"] = labels

        unique_labels = sorted(df_result["cluster"].unique())
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        cmap = plt.colormaps.get_cmap("tab20")
        norm = mcolors.Normalize(vmin=0, vmax=max(len(unique_labels) - 1, 1))
        colors = [cmap(norm(i)) for i in range(len(unique_labels))]

        df_result["color_idx"] = df_result["cluster"].map(label_to_index)

        df_result_sorted = df_result.copy()
        df_result_sorted["sort_key"] = df_result_sorted["cluster"].apply(lambda x: 999 if x == -1 else x)
        df_result_sorted = df_result_sorted.sort_values("sort_key")

        n_clusters, _ = HDBSCANWorkflow.cluster_counts(labels)
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

        plt.ylim(cfg.fixed_ylim[0], cfg.fixed_ylim[1])

        plt.title(title, fontsize=17)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()