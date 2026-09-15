import os
import sys
import numpy as np
import pandas as pd
try:
    import hdbscan
    HDBSCAN_CLS = hdbscan.HDBSCAN
except ImportError:
    from sklearn.cluster import HDBSCAN as HDBSCAN_CLS
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from location_metadata import CLUSTER_LOCATIONS, get_location_name
except ImportError:
    from location_metadata import CLUSTER_LOCATIONS, get_location_name



# ============================================================
# FINAL HDBSCAN SOURCE-REGION ANALYSIS
# ============================================================

INPUT_FILE = (
    "features/SPATIAL_SOURCE_CANDIDATES_2025.csv"
)

OUTPUT_CLUSTERS = (
    "features/HCHO_SOURCE_CLUSTERS_HDBSCAN_2025.csv"
)

OUTPUT_SUMMARY = (
    "features/HCHO_SOURCE_REGION_SUMMARY_HDBSCAN_2025.csv"
)

OUTPUT_MAP = (
    "features/HCHO_SOURCE_REGIONS_HDBSCAN_2025.png"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 5


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("FINAL HDBSCAN SOURCE-REGION ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(
    "Spatial candidates:",
    len(df)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "latitude",
    "longitude",
    "active_months",
    "mean_HCHO",
    "max_HCHO",
    "total_fire_count",
    "max_fire_count",
    "total_FRP",
    "max_FRP",
    "mean_source_score",
    "max_source_score"
]


missing = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing:

    raise ValueError(
        "Missing columns: "
        + str(missing)
    )


print("\nInput columns:")
print(df.columns.tolist())


# ============================================================
# CLEAN DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=[
        "latitude",
        "longitude"
    ]
).copy()

df.reset_index(
    drop=True,
    inplace=True
)


print(
    "\nValid spatial candidates:",
    len(df)
)


# ============================================================
# CONVERT LAT/LON TO APPROXIMATE KM
# ============================================================
#
# HDBSCAN uses Euclidean distance.
#
# Latitude and longitude are therefore converted approximately
# into kilometres before clustering.
# ============================================================

lat = df[
    "latitude"
].to_numpy(
    dtype=float
)

lon = df[
    "longitude"
].to_numpy(
    dtype=float
)


lat_km = (
    lat * 111.32
)


lon_km = (
    lon
    * 111.32
    * np.cos(
        np.radians(lat)
    )
)


X = np.column_stack(
    [
        lon_km,
        lat_km
    ]
)


# ============================================================
# HDBSCAN
# ============================================================

print(
    "\n========== HDBSCAN =========="
)

print(
    "min_cluster_size =",
    MIN_CLUSTER_SIZE
)

print(
    "min_samples       =",
    MIN_SAMPLES
)


clusterer = HDBSCAN_CLS(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    cluster_selection_method="eom"
)


labels = clusterer.fit_predict(
    X
)



df["cluster_id"] = labels


# ============================================================
# HDBSCAN PROBABILITY
# ============================================================

df["cluster_probability"] = (
    clusterer.probabilities_
)


# ============================================================
# BASIC RESULTS
# ============================================================

cluster_ids = sorted(
    [
        cluster
        for cluster in np.unique(labels)
        if cluster != -1
    ]
)


n_clusters = len(
    cluster_ids
)


noise_mask = (
    labels == -1
)


noise_points = int(
    noise_mask.sum()
)


clustered_points = (
    len(df)
    - noise_points
)


noise_percent = (
    100
    * noise_points
    / len(df)
)


print(
    "\n========== HDBSCAN RESULTS =========="
)

print(
    "Total candidates:",
    len(df)
)

print(
    "Clusters:",
    n_clusters
)

print(
    "Clustered cells:",
    clustered_points
)

print(
    "Noise cells:",
    noise_points
)

print(
    "Noise percentage:",
    f"{noise_percent:.2f}%"
)


persistence = getattr(clusterer, 'cluster_persistence_', [1.0] * len(cluster_ids))


print(
    "\n========== CLUSTER PERSISTENCE =========="
)


for cluster_id, value in zip(
    cluster_ids,
    persistence
):

    print(
        f"Cluster {cluster_id}: "
        f"{float(value):.3f}"
    )



# ============================================================
# CLUSTER SUMMARY
# ============================================================

summary_rows = []


for cluster_id in cluster_ids:

    subset = df[
        df["cluster_id"]
        == cluster_id
    ].copy()


    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    persistence_index = (
        cluster_ids.index(
            cluster_id
        )
    )

    cluster_persistence = float(
        persistence[
            persistence_index
        ]
    )


    # --------------------------------------------------------
    # Basic spatial information
    # --------------------------------------------------------

    mean_latitude = (
        subset["latitude"].mean()
    )

    mean_longitude = (
        subset["longitude"].mean()
    )


    min_latitude = (
        subset["latitude"].min()
    )

    max_latitude = (
        subset["latitude"].max()
    )

    min_longitude = (
        subset["longitude"].min()
    )

    max_longitude = (
        subset["longitude"].max()
    )


    # --------------------------------------------------------
    # Cluster probability
    # --------------------------------------------------------

    mean_probability = (
        subset[
            "cluster_probability"
        ].mean()
    )

    max_probability = (
        subset[
            "cluster_probability"
        ].max()
    )


    # --------------------------------------------------------
    # Emission characteristics
    # --------------------------------------------------------

    mean_hcho = (
        subset["mean_HCHO"].mean()
    )

    max_hcho = (
        subset["max_HCHO"].max()
    )


    mean_source_score = (
        subset[
            "mean_source_score"
        ].mean()
    )

    max_source_score = (
        subset[
            "max_source_score"
        ].max()
    )


    total_fire = (
        subset[
            "total_fire_count"
        ].sum()
    )

    max_fire = (
        subset[
            "max_fire_count"
        ].max()
    )


    total_frp = (
        subset[
            "total_FRP"
        ].sum()
    )

    max_frp = (
        subset[
            "max_FRP"
        ].max()
    )


    # --------------------------------------------------------
    # Temporal characteristics
    # --------------------------------------------------------

    mean_active_months = (
        subset[
            "active_months"
        ].mean()
    )

    max_active_months = (
        subset[
            "active_months"
        ].max()
    )


    # --------------------------------------------------------
    # Number of cells
    # --------------------------------------------------------

    cells = len(
        subset
    )


    # --------------------------------------------------------
    # Priority score
    # --------------------------------------------------------
    #
    # This is a ranking index, NOT a probability.
    #
    # It combines:
    #   - source score
    #   - fire activity
    #   - FRP
    #   - HCHO
    #   - temporal persistence
    #   - HDBSCAN persistence
    #
    # Each component is normalized across clusters.
    # --------------------------------------------------------

    summary_rows.append({

        "cluster_id":
            cluster_id,

        "cells":
            cells,

        "mean_latitude":
            mean_latitude,

        "mean_longitude":
            mean_longitude,

        "min_latitude":
            min_latitude,

        "max_latitude":
            max_latitude,

        "min_longitude":
            min_longitude,

        "max_longitude":
            max_longitude,

        "mean_cluster_probability":
            mean_probability,

        "max_cluster_probability":
            max_probability,

        "mean_HCHO":
            mean_hcho,

        "max_HCHO":
            max_hcho,

        "mean_source_score":
            mean_source_score,

        "max_source_score":
            max_source_score,

        "total_fire_count":
            total_fire,

        "max_fire_count":
            max_fire,

        "total_FRP":
            total_frp,

        "max_FRP":
            max_frp,

        "mean_active_months":
            mean_active_months,

        "max_active_months":
            max_active_months,

        "cluster_persistence":
            cluster_persistence
    })


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def normalize(series):

    minimum = series.min()

    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            1.0,
            index=series.index
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# NORMALIZED COMPONENTS
# ============================================================

summary[
    "score_source"
] = normalize(
    summary[
        "mean_source_score"
    ]
)


summary[
    "score_hcho"
] = normalize(
    summary[
        "mean_HCHO"
    ]
)


summary[
    "score_fire"
] = normalize(
    np.log1p(
        summary[
            "total_fire_count"
        ]
    )
)


summary[
    "score_frp"
] = normalize(
    np.log1p(
        summary[
            "total_FRP"
        ]
    )
)


summary[
    "score_temporal"
] = normalize(
    summary[
        "mean_active_months"
    ]
)


summary[
    "score_persistence"
] = normalize(
    summary[
        "cluster_persistence"
    ]
)


# ============================================================
# REGION PRIORITY SCORE
# ============================================================
#
# Equal weighting prevents one variable from dominating.
# This is an operational ranking metric, not a causal measure.
# ============================================================

summary[
    "region_priority_score"
] = (

    0.25
    * summary["score_source"]

    + 0.20
    * summary["score_hcho"]

    + 0.20
    * summary["score_fire"]

    + 0.20
    * summary["score_frp"]

    + 0.10
    * summary["score_temporal"]

    + 0.05
    * summary["score_persistence"]
)


# ============================================================
# RANK SOURCE REGIONS
# ============================================================

summary = summary.sort_values(
    "region_priority_score",
    ascending=False
).reset_index(
    drop=True
)


summary[
    "source_region_rank"
] = (
    summary.index + 1
)


# ============================================================
# REORDER IMPORTANT COLUMNS
# ============================================================

priority_columns = [

    "source_region_rank",
    "cluster_id",
    "cells",

    "mean_latitude",
    "mean_longitude",

    "min_latitude",
    "max_latitude",

    "min_longitude",
    "max_longitude",

    "mean_cluster_probability",
    "max_cluster_probability",

    "mean_HCHO",
    "max_HCHO",

    "mean_source_score",
    "max_source_score",

    "total_fire_count",
    "max_fire_count",

    "total_FRP",
    "max_FRP",

    "mean_active_months",
    "max_active_months",

    "cluster_persistence",

    "region_priority_score"
]


summary = summary[
    priority_columns
]


# ============================================================
# SAVE POINT-LEVEL DATA
# ============================================================

os.makedirs(
    "features",
    exist_ok=True
)


df.to_csv(
    OUTPUT_CLUSTERS,
    index=False
)


# ============================================================
# SAVE REGION SUMMARY
# ============================================================

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n========== SOURCE REGION SUMMARY =========="
)

print(
    summary[
        [
            "source_region_rank",
            "cluster_id",
            "cells",
            "mean_latitude",
            "mean_longitude",
            "mean_HCHO",
            "mean_source_score",
            "total_fire_count",
            "total_FRP",
            "mean_active_months",
            "cluster_persistence",
            "region_priority_score"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# TOP SOURCE REGIONS
# ============================================================

print(
    "\n========== TOP SOURCE REGIONS =========="
)

print(
    summary[
        [
            "source_region_rank",
            "cluster_id",
            "cells",
            "mean_latitude",
            "mean_longitude",
            "mean_HCHO",
            "mean_source_score",
            "total_fire_count",
            "total_FRP",
            "mean_active_months",
            "cluster_persistence",
            "region_priority_score"
        ]
    ].head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# CREATE MAP
# ============================================================

print(
    "\n========== CREATING MAP =========="
)


fig, ax = plt.subplots(
    figsize=(12, 9)
)


# ------------------------------------------------------------
# Noise
# ------------------------------------------------------------

if np.any(noise_mask):

    ax.scatter(

        df.loc[
            noise_mask,
            "longitude"
        ],

        df.loc[
            noise_mask,
            "latitude"
        ],

        s=2,

        alpha=0.15,

        label="Noise"
    )


# ------------------------------------------------------------
# Clusters
# ------------------------------------------------------------

for cluster_id in cluster_ids:

    mask = (
        df["cluster_id"]
        == cluster_id
    )

    ax.scatter(

        df.loc[
            mask,
            "longitude"
        ],

        df.loc[
            mask,
            "latitude"
        ],

        s=5,

        alpha=0.55
    )


# ------------------------------------------------------------
# Label top regions
# ------------------------------------------------------------

top_regions = summary.head(
    min(14, len(summary))
)


for _, row in top_regions.iterrows():
    rank_num = int(row['source_region_rank'])
    loc_lbl = f"#{rank_num} {get_location_name(row['cluster_id'], 'short')}"

    ax.annotate(
        loc_lbl,
        (
            row["mean_longitude"],
            row["mean_latitude"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="#CBD5E1", linewidth=0.5)
    )


ax.set_xlabel(
    "Longitude (°E)",
    fontsize=11,
    fontweight="bold"
)

ax.set_ylabel(
    "Latitude (°N)",
    fontsize=11,
    fontweight="bold"
)

ax.set_title(
    "Final HDBSCAN Source Regions & Clustered Hotspot Locations — 2025",
    fontsize=14,
    fontweight="bold",
    pad=12
)


ax.set_xlim(
    68,
    97.5
)

ax.set_ylim(
    6,
    37.5
)

ax.grid(
    True,
    alpha=0.2
)


plt.tight_layout()


plt.savefig(
    OUTPUT_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\nSaved point-level clusters:"
)

print(
    OUTPUT_CLUSTERS
)

print(
    "\nSaved source-region summary:"
)

print(
    OUTPUT_SUMMARY
)

print(
    "\nSaved map:"
)

print(
    OUTPUT_MAP
)


print(
    "\n" + "=" * 60
)

print(
    "FINAL HDBSCAN SOURCE-REGION ANALYSIS COMPLETE"
)

print(
    "=" * 60
)