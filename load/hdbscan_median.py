import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import math
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from itertools import product
from pathlib import Path

# ======================================
# 🔧 USER CONFIGURATION
# ======================================
use_month_feature = False
month_encoding = "cyclical"
use_pca = True
use_grid_search = False
cluster_selection_method = "leaf"

manual_params = {
    "min_cluster_size": 350,
    "min_samples": 250,
    "cluster_selection_epsilon": 0.08
}

metric = "manhattan"
sample_fraction = 0.01


# save_dir = f"results"

script_dir = Path(__file__).resolve().parent
save_dir = script_dir / "results"
save_dir.mkdir(exist_ok=True)


os.makedirs(save_dir, exist_ok=True)
log_csv_path = os.path.join(save_dir, "hdbscan_log.csv")

# ======================================
# 📦 LOAD DATA
# ======================================
df = pd.read_parquet("/mnt/raid_data/data/monthly/lf_2021-11-04.parquet")

df["month"] = df["month"].astype(str)
df["month_abbr"] = df["month"].str.extract(r"-(\w{3})")
month_map = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}
df["month_numeric"] = df["month_abbr"].map(month_map)
if df["month_numeric"].isnull().any():
    raise ValueError(f"Unmapped months: {df[df['month_numeric'].isnull()]['month_abbr'].unique()}")

if month_encoding == "cyclical":
    radians = (df["month_numeric"] - 1) / 11 * 2 * math.pi
    df["month_sin"] = np.sin(radians)
    df["month_cos"] = np.cos(radians)
elif month_encoding == "seasonal":
    df["season"] = df["month_numeric"].map({
        12: 0, 1: 0, 2: 0,
        3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3
    })

# ======================================
# 🧼 SELECT FEATURES
# ======================================
if not use_month_feature:
    features = ["attenuation_dB", "carrier_frequency_kHz"]
elif month_encoding == "cyclical":
    features = ["attenuation_dB", "carrier_frequency_kHz", "month_sin", "month_cos"]
elif month_encoding == "numeric":
    features = ["attenuation_dB", "carrier_frequency_kHz", "month_numeric"]
elif month_encoding == "seasonal":
    features = ["attenuation_dB", "carrier_frequency_kHz", "season"]
else:
    raise ValueError(f"Invalid month_encoding: {month_encoding}")

# ======================================
# 📊 SAMPLE & SCALE DATA
# ======================================
df_sampled = df.groupby("location_tag").apply(
    lambda x: x.sample(frac=sample_fraction, random_state=42),
    include_groups=False
)
X = df_sampled[features].dropna()
X_scaled = StandardScaler().fit_transform(X)

if use_pca:
    pca = PCA(n_components=2)
    X_final = pca.fit_transform(X_scaled)
    pca_suffix = "_pca2d"
else:
    X_final = X_scaled
    pca_suffix = ""

# ======================================
# 🔁 PARAMETER CONFIG
# ======================================
if use_grid_search:
    param_grid = list(product(
        [300],
        [200],
        [0.075, 0.08]
    ))
else:
    param_grid = [(
        manual_params["min_cluster_size"],
        manual_params["min_samples"],
        manual_params["cluster_selection_epsilon"]
    )]

# ======================================
# 🚀 RUN HDBSCAN
# ======================================
for mc, ms, eps in param_grid:
    print(f"\n🔄 Running HDBSCAN with mc={mc}, ms={ms}, eps={eps}")
    suffix = f"mc{mc}_ms{ms}_eps{int(eps*1000)}_{cluster_selection_method}_{metric}{pca_suffix}"
    if use_month_feature:
        suffix += f"_{month_encoding}month"

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mc,
        min_samples=ms,
        cluster_selection_epsilon=eps,
        cluster_selection_method=cluster_selection_method,
        metric=metric
    )
    labels = clusterer.fit_predict(X_final)

    df_result = X.copy()
    df_result["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    # ======================================
    # 📈 PLOT RESULTS + SMOOTHED MEDIAN OVERLAY
    # ======================================
    unique_labels = sorted(df_result["cluster"].unique())
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    cmap = plt.colormaps.get_cmap("tab20")
    norm = mcolors.Normalize(vmin=0, vmax=len(unique_labels) - 1)
    colors = [cmap(norm(i)) for i in range(len(unique_labels))]
    df_result["color_idx"] = df_result["cluster"].map(label_to_index)
    df_result_sorted = df_result.copy()
    df_result_sorted["sort_key"] = df_result_sorted["cluster"].apply(lambda x: 999 if x == -1 else x)
    df_result_sorted = df_result_sorted.sort_values("sort_key")

    alpha = 0.25 if n_clusters <= 10 else 0.15

    plt.figure(figsize=(14, 10))
    plt.scatter(
        df_sampled.loc[df_result_sorted.index, "carrier_frequency_kHz"],
        df_sampled.loc[df_result_sorted.index, "attenuation_dB"],
        c=df_result_sorted["color_idx"],
        cmap=cmap,
        s=3,
        alpha=alpha,
        edgecolors='none'
    )

    # ➕ Rolling median overlay
    med_df = df_sampled.loc[df_result_sorted.index].groupby("carrier_frequency_kHz")["attenuation_dB"].median().reset_index()
    med_df = med_df.sort_values("carrier_frequency_kHz")
    med_df["smoothed_median"] = med_df["attenuation_dB"].rolling(window=7, center=True).median()

    plt.plot(
        med_df["carrier_frequency_kHz"],
        med_df["smoothed_median"],
        color="black",
        linewidth=2,
        linestyle="-",
        label="Median"
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='none', label=str(label),
               markerfacecolor=colors[i], markersize=6)
        for i, label in enumerate(unique_labels)
    ]
    legend_elements.append(
        Line2D([0], [0], color="black", lw=2, label="Median")  # median curve line
    )

    # plt.legend(
    #     handles=legend_elements,
    #     title="Cluster label",
    #     bbox_to_anchor=(1.025, 1),
    #     loc="upper left",
    #     borderaxespad=0.0
    # )

    plt.legend(
        handles=legend_elements,
        title="Cluster label",
        title_fontsize=15,
        fontsize=15,
        bbox_to_anchor=(1.025, 1),
        loc="upper left",
        borderaxespad=0.0
    )

    plt.xlabel("Sub-Carrier Frequency in kHz", fontsize=17, labelpad=10)
    plt.ylabel("Attenuation in dB", fontsize=17, labelpad=10)
    plt.ylim(-60, 100)
    plt.xlim(-30, 530)

    pca_info = f"PCA({pca.n_components_}D)" if use_pca else "No PCA"
    month_info = f", {month_encoding.capitalize()} Month" if use_month_feature else ""
    title = (
        f"mc={mc}, ms={ms}, eps={eps} | "
        f"method={cluster_selection_method}, metric={metric}, {pca_info}{month_info}"
    )
    plt.title(title, fontsize=17)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/hdbscan_clusters_{suffix}_with_rollingmedian_new_bifger_font.png", dpi=500)

    # plt.savefig(f"{save_dir}/SVG Plots/hdbscan_clusters_{suffix}_with_rollingmedian_new.svg", format="svg")

    plt.close()

    print(f"✅ Done with mc={mc}, ms={ms}, eps={eps} | Clusters: {n_clusters}, Noise: {n_noise}")
