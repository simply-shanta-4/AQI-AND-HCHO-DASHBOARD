import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = "features/HCHO_FIRE_WIND_2025.csv"

CLUSTER_FILE = (
    "features/HCHO_SOURCE_CLUSTERS_HDBSCAN_2025.csv"
)

SUMMARY_FILE = (
    "features/HCHO_SOURCE_REGION_SUMMARY_HDBSCAN_2025.csv"
)

OUTPUT_FILE = (
    "features/HCHO_SOURCE_REGIONS_WIND_VALIDATED_2025.csv"
)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 60)
print("OPTIMIZED WIND TRANSPORT VALIDATION")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading feature dataset...")

features = pd.read_csv(FEATURE_FILE)

print("Feature rows:", len(features))


print("\nLoading HDBSCAN spatial candidates...")

clusters = pd.read_csv(CLUSTER_FILE)

print("Spatial candidate rows:", len(clusters))


print("\nLoading HDBSCAN region summary...")

summary = pd.read_csv(SUMMARY_FILE)

print("Source regions:", len(summary))


# ============================================================
# CHECK COLUMNS
# ============================================================

required_features = [
    "month",
    "latitude",
    "longitude",
    "HCHO",
    "fire_count",
    "FRP",
    "u10",
    "v10",
    "wind_speed",
    "wind_direction"
]

required_clusters = [
    "latitude",
    "longitude",
    "cluster_id"
]


missing_features = [
    c for c in required_features
    if c not in features.columns
]

missing_clusters = [
    c for c in required_clusters
    if c not in clusters.columns
]


if missing_features:
    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )


if missing_clusters:
    raise ValueError(
        f"Missing cluster columns: {missing_clusters}"
    )


# ============================================================
# CLEAN
# ============================================================

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

clusters = clusters.replace(
    [np.inf, -np.inf],
    np.nan
)


features = features.dropna(
    subset=[
        "latitude",
        "longitude",
        "u10",
        "v10",
        "wind_speed",
        "wind_direction"
    ]
).copy()


clusters = clusters.dropna(
    subset=[
        "latitude",
        "longitude",
        "cluster_id"
    ]
).copy()


# ============================================================
# REMOVE HDBSCAN NOISE
# ============================================================

clusters = clusters[
    clusters["cluster_id"] != -1
].copy()


print(
    "\nNon-noise HDBSCAN cells:",
    len(clusters)
)

print(
    "HDBSCAN regions:",
    clusters["cluster_id"].nunique()
)


# ============================================================
# CREATE UNIQUE HCHO SPATIAL GRID
# ============================================================
#
# The feature file contains the same spatial grid repeated
# for every month.
#
# Therefore we DO NOT need to process 1.22 million spatial
# coordinates.
#
# We first extract the unique latitude/longitude grid.
#
# ============================================================

print("\nPreparing unique HCHO spatial grid...")


grid = features[
    ["latitude", "longitude"]
].drop_duplicates().reset_index(
    drop=True
)


print(
    "Unique HCHO grid cells:",
    len(grid)
)


# ============================================================
# COORDINATE TRANSFORMATION
# ============================================================
#
# India covers a limited latitude range.
#
# For nearest-neighbour matching, scale longitude by
# cos(mean latitude) so longitude and latitude have
# approximately comparable physical distances.
#
# ============================================================

mean_latitude = (
    features["latitude"].mean()
)


lon_scale = np.cos(
    np.radians(mean_latitude)
)


grid_xy = np.column_stack(
    [
        grid["latitude"].to_numpy(
            dtype=float
        ),

        grid["longitude"].to_numpy(
            dtype=float
        ) * lon_scale
    ]
)


candidate_xy = np.column_stack(
    [
        clusters["latitude"].to_numpy(
            dtype=float
        ),

        clusters["longitude"].to_numpy(
            dtype=float
        ) * lon_scale
    ]
)


# ============================================================
# BUILD KD-TREE
# ============================================================

print("\nBuilding spatial KD-tree...")

tree = cKDTree(
    grid_xy
)


# ============================================================
# MATCH HDBSCAN CELLS TO HCHO GRID
# ============================================================
#
# ONLY ~20,000 queries now.
#
# This replaces the previous ~24.5 BILLION comparisons.
#
# ============================================================

print(
    "Matching HDBSCAN cells to nearest HCHO grid cells..."
)


distances, nearest_indices = tree.query(
    candidate_xy,
    k=1
)


clusters[
    "grid_index"
] = nearest_indices


clusters[
    "grid_distance_deg"
] = distances


# Approximate km distance

clusters[
    "grid_distance_km"
] = (
    distances
    * 111.32
)


print(
    "Maximum matching distance:",
    f"{clusters['grid_distance_km'].max():.3f} km"
)

print(
    "Mean matching distance:",
    f"{clusters['grid_distance_km'].mean():.3f} km"
)


# ============================================================
# CREATE GRID INDEX IN FEATURE DATA
# ============================================================

grid[
    "grid_index"
] = np.arange(
    len(grid)
)


# ============================================================
# SELECT ONLY HCHO GRID CELLS USED BY HDBSCAN
# ============================================================

used_grid_indices = np.unique(
    nearest_indices
)


print(
    "\nHCHO grid cells required for wind analysis:",
    len(used_grid_indices)
)


# ============================================================
# FILTER FEATURE DATA
# ============================================================

print(
    "\nSelecting monthly observations for source regions..."
)


selected_grid = grid[
    grid["grid_index"].isin(
        used_grid_indices
    )
].copy()


# ============================================================
# MERGE GRID INDEX WITH FEATURES
# ============================================================
#
# Instead of spatially matching 1.22 million records,
# match using the exact latitude/longitude grid.
#
# ============================================================

features_selected = features.merge(
    selected_grid[
        [
            "latitude",
            "longitude",
            "grid_index"
        ]
    ],
    on=[
        "latitude",
        "longitude"
    ],
    how="inner"
)


print(
    "Selected monthly observations:",
    len(features_selected)
)


# ============================================================
# MAP GRID CELL -> HDBSCAN CLUSTER
# ============================================================

grid_cluster_map = clusters[
    [
        "grid_index",
        "cluster_id"
    ]
].drop_duplicates()


# If multiple HDBSCAN candidate cells map to the same HCHO
# grid cell and belong to different clusters, keep the
# closest candidate assignment.

cluster_distance_map = clusters[
    [
        "grid_index",
        "cluster_id",
        "grid_distance_km"
    ]
].sort_values(
    "grid_distance_km"
)


cluster_distance_map = (
    cluster_distance_map
    .drop_duplicates(
        subset=["grid_index"],
        keep="first"
    )
)


features_selected = features_selected.merge(
    cluster_distance_map[
        [
            "grid_index",
            "cluster_id"
        ]
    ],
    on="grid_index",
    how="inner"
)


print(
    "Observations assigned to HDBSCAN regions:",
    len(features_selected)
)


# ============================================================
# METEOROLOGICAL TRANSPORT DIRECTION
# ============================================================
#
# ERA5 wind_direction represents the direction FROM which
# the wind originates.
#
# Transport direction is therefore:
#
#     (wind_direction + 180) % 360
#
# ============================================================

features_selected[
    "transport_direction"
] = (
    features_selected[
        "wind_direction"
    ]
    + 180.0
) % 360.0


# ============================================================
# VECTOR COMPONENTS
# ============================================================

features_selected[
    "transport_u"
] = features_selected[
    "u10"
]

features_selected[
    "transport_v"
] = features_selected[
    "v10"
]


# ============================================================
# WIND STATISTICS BY REGION
# ============================================================

print(
    "\nCalculating regional wind statistics..."
)


wind_rows = []


for cluster_id, group in (
    features_selected
    .groupby("cluster_id")
):

    # --------------------------------------------------------
    # Mean vector
    # --------------------------------------------------------

    mean_u = group[
        "transport_u"
    ].mean()

    mean_v = group[
        "transport_v"
    ].mean()


    # --------------------------------------------------------
    # Resultant vector magnitude
    # --------------------------------------------------------

    resultant_speed = np.sqrt(
        mean_u ** 2
        + mean_v ** 2
    )


    # --------------------------------------------------------
    # Mean wind speed
    # --------------------------------------------------------

    mean_wind_speed = group[
        "wind_speed"
    ].mean()


    max_wind_speed = group[
        "wind_speed"
    ].max()


    # --------------------------------------------------------
    # Mean transport direction
    # --------------------------------------------------------
    #
    # atan2(EAST, NORTH)
    #
    # gives a meteorological-style clockwise direction
    # measured from north.
    #
    # --------------------------------------------------------

    transport_direction = (
        np.degrees(
            np.arctan2(
                mean_u,
                mean_v
            )
        )
        + 360.0
    ) % 360.0


    # --------------------------------------------------------
    # Directional consistency
    # --------------------------------------------------------
    #
    # Resultant length:
    #
    # R = resultant vector / mean scalar speed
    #
    # close to 1:
    #   wind is directionally consistent
    #
    # close to 0:
    #   wind direction varies strongly
    #
    # --------------------------------------------------------

    if mean_wind_speed > 0:

        directional_consistency = (
            resultant_speed
            / mean_wind_speed
        )

    else:

        directional_consistency = 0.0


    directional_consistency = float(
        np.clip(
            directional_consistency,
            0.0,
            1.0
        )
    )


    # --------------------------------------------------------
    # Number of monthly observations
    # --------------------------------------------------------

    observations = len(
        group
    )


    # --------------------------------------------------------
    # Months
    # --------------------------------------------------------

    active_months = (
        group[
            "month"
        ].nunique()
    )


    # --------------------------------------------------------
    # HCHO
    # --------------------------------------------------------

    mean_hcho = group[
        "HCHO"
    ].mean()

    max_hcho = group[
        "HCHO"
    ].max()


    # --------------------------------------------------------
    # Fire
    # --------------------------------------------------------

    total_fire = group[
        "fire_count"
    ].sum()

    max_fire = group[
        "fire_count"
    ].max()


    # --------------------------------------------------------
    # FRP
    # --------------------------------------------------------

    total_frp = group[
        "FRP"
    ].sum()

    max_frp = group[
        "FRP"
    ].max()


    wind_rows.append({

        "cluster_id":
            int(cluster_id),

        "wind_observations":
            observations,

        "wind_active_months":
            active_months,

        "wind_mean_HCHO":
            mean_hcho,

        "wind_max_HCHO":
            max_hcho,

        "wind_total_fire_count":
            total_fire,

        "wind_total_FRP":
            total_frp,

        "mean_wind_speed":
            mean_wind_speed,

        "max_wind_speed":
            max_wind_speed,

        "mean_u10":
            mean_u,

        "mean_v10":
            mean_v,

        "resultant_wind_speed":
            resultant_speed,

        "transport_direction":
            transport_direction,

        "directional_consistency":
            directional_consistency
    })


wind_summary = pd.DataFrame(
    wind_rows
)


# ============================================================
# MERGE WITH HDBSCAN REGION SUMMARY
# ============================================================

print(
    "\nMerging wind statistics with HDBSCAN regions..."
)


result = summary.merge(
    wind_summary,
    on="cluster_id",
    how="left"
)


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def minmax(series):

    series = series.astype(float)

    minimum = series.min()
    maximum = series.max()

    if (
        not np.isfinite(minimum)
        or
        not np.isfinite(maximum)
        or
        maximum == minimum
    ):

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
# WIND COMPONENT SCORES
# ============================================================

result[
    "wind_speed_score"
] = minmax(
    result[
        "resultant_wind_speed"
    ]
)


result[
    "wind_consistency_score"
] = (
    result[
        "directional_consistency"
    ]
    .clip(
        0,
        1
    )
)


# ============================================================
# WIND TRANSPORT SCORE
# ============================================================

result[
    "wind_transport_score"
] = (

    0.50
    * result[
        "wind_speed_score"
    ]

    +

    0.50
    * result[
        "wind_consistency_score"
    ]
)


# ============================================================
# WIND-VALIDATED PRIORITY
# ============================================================
#
# IMPORTANT:
#
# We retain the original HDBSCAN region priority.
#
# Wind contributes only 20%.
#
# This prevents the wind diagnostic from overpowering
# the source evidence.
#
# ============================================================

result[
    "wind_validated_score"
] = (

    0.80
    * result[
        "region_priority_score"
    ]

    +

    0.20
    * result[
        "wind_transport_score"
    ]
)


# ============================================================
# RANK
# ============================================================

result = result.sort_values(
    "wind_validated_score",
    ascending=False
).reset_index(
    drop=True
)


result[
    "wind_validated_rank"
] = (
    result.index + 1
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "features",
    exist_ok=True
)


result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# TOP RESULTS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "WIND-VALIDATED SOURCE REGIONS"
)

print(
    "=" * 60
)


display_columns = [

    "wind_validated_rank",

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

    "mean_wind_speed",

    "resultant_wind_speed",

    "transport_direction",

    "directional_consistency",

    "wind_transport_score",

    "wind_validated_score"
]


print(
    result[
        display_columns
    ].head(20).to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# WIND DIRECTION INTERPRETATION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "WIND DIRECTION INTERPRETATION"
)

print(
    "=" * 60
)

print(
    "transport_direction = direction toward which air moves"
)

print(
    "wind_direction = direction from which wind originates"
)


# ============================================================
# SAVE
# ============================================================

print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


print(
    "\n" + "=" * 60
)

print(
    "WIND TRANSPORT VALIDATION COMPLETE"
)

print(
    "=" * 60
)