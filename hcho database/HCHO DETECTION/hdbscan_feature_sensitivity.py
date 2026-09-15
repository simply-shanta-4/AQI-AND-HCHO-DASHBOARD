import os
import numpy as np
import pandas as pd
import hdbscan

from sklearn.preprocessing import RobustScaler


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "features/SPATIAL_SOURCE_CANDIDATES_2025.csv"
)

OUTPUT_FILE = (
    "features/HDBSCAN_FEATURE_SENSITIVITY_2025.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("HDBSCAN FEATURE-SPACE SENSITIVITY ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(
    "Spatial candidates:",
    len(df)
)


# ============================================================
# REQUIRED VARIABLES
# ============================================================

feature_columns = [
    "latitude",
    "longitude",
    "mean_HCHO",
    "total_fire_count",
    "total_FRP"
]

missing = [
    column
    for column in feature_columns
    if column not in df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# CLEAN
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=feature_columns
).copy()

print(
    "Valid candidates:",
    len(df)
)


# ============================================================
# TRANSFORM SKEWED VARIABLES
# ============================================================
#
# Fire count and FRP are strongly right-skewed.
# Log transformation prevents a few extreme values from
# dominating the clustering.
#
# HCHO is also transformed after ensuring positivity.
# ============================================================

df["log_HCHO"] = np.log1p(
    np.maximum(
        df["mean_HCHO"],
        0
    )
)

df["log_fire"] = np.log1p(
    np.maximum(
        df["total_fire_count"],
        0
    )
)

df["log_FRP"] = np.log1p(
    np.maximum(
        df["total_FRP"],
        0
    )
)


# ============================================================
# CONVERT LAT/LON TO APPROXIMATE KM
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


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

X_raw = np.column_stack(
    [
        lon_km,
        lat_km,
        df["log_HCHO"].to_numpy(),
        df["log_fire"].to_numpy(),
        df["log_FRP"].to_numpy()
    ]
)


# ============================================================
# ROBUST SCALING
# ============================================================
#
# RobustScaler uses median and IQR and is less sensitive to
# extreme fire/FRP observations than ordinary standardization.
# ============================================================

scaler = RobustScaler()

X = scaler.fit_transform(
    X_raw
)


# ============================================================
# PARAMETER GRID
# ============================================================

min_cluster_sizes = [
    10,
    20,
    30,
    50,
    75,
    100
]

min_samples_values = [
    5,
    10,
    20
]


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

results = []


for min_cluster_size in min_cluster_sizes:

    for min_samples in min_samples_values:

        print(
            "\nTesting:"
        )

        print(
            "min_cluster_size =",
            min_cluster_size
        )

        print(
            "min_samples =",
            min_samples
        )


        clusterer = hdbscan.HDBSCAN(

            min_cluster_size=(
                min_cluster_size
            ),

            min_samples=(
                min_samples
            ),

            metric="euclidean",

            cluster_selection_method="eom"
        )


        labels = clusterer.fit_predict(
            X
        )


        # ----------------------------------------------------
        # CLUSTER LABELS
        # ----------------------------------------------------

        cluster_labels = sorted(
            [
                label
                for label in np.unique(labels)
                if label != -1
            ]
        )


        n_clusters = len(
            cluster_labels
        )


        # ----------------------------------------------------
        # NOISE
        # ----------------------------------------------------

        noise_points = int(
            np.sum(
                labels == -1
            )
        )

        total_points = len(
            labels
        )

        noise_percent = (
            100
            * noise_points
            / total_points
        )


        clustered_points = (
            total_points
            - noise_points
        )


        # ----------------------------------------------------
        # CLUSTER SIZES
        # ----------------------------------------------------

        if n_clusters > 0:

            cluster_sizes = [

                int(
                    np.sum(
                        labels == label
                    )
                )

                for label in cluster_labels
            ]

            largest_cluster = max(
                cluster_sizes
            )

            smallest_cluster = min(
                cluster_sizes
            )

            median_cluster_size = float(
                np.median(
                    cluster_sizes
                )
            )

        else:

            largest_cluster = 0

            smallest_cluster = 0

            median_cluster_size = 0


        # ----------------------------------------------------
        # CLUSTER PERSISTENCE
        # ----------------------------------------------------

        persistence = (
            clusterer.cluster_persistence_
        )


        if len(persistence) > 0:

            mean_persistence = float(
                np.mean(
                    persistence
                )
            )

            max_persistence = float(
                np.max(
                    persistence
                )
            )

        else:

            mean_persistence = 0.0

            max_persistence = 0.0


        # ----------------------------------------------------
        # LARGEST CLUSTER FRACTION
        # ----------------------------------------------------

        if total_points > 0:

            largest_cluster_fraction = (
                largest_cluster
                / total_points
            )

        else:

            largest_cluster_fraction = 0.0


        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        results.append({

            "min_cluster_size":
                min_cluster_size,

            "min_samples":
                min_samples,

            "clusters":
                n_clusters,

            "noise_points":
                noise_points,

            "noise_percent":
                noise_percent,

            "clustered_points":
                clustered_points,

            "largest_cluster":
                largest_cluster,

            "largest_cluster_fraction":
                largest_cluster_fraction,

            "median_cluster_size":
                median_cluster_size,

            "smallest_cluster":
                smallest_cluster,

            "mean_persistence":
                mean_persistence,

            "max_persistence":
                max_persistence
        })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "features",
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "HDBSCAN FEATURE-SPACE RESULTS"
)

print(
    "=" * 60
)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# FIND HIGH-PERSISTENCE SOLUTIONS
# ============================================================

print(
    "\n========== HIGH-PERSISTENCE SOLUTIONS =========="
)

best_results = (
    results_df
    .sort_values(
        [
            "mean_persistence",
            "largest_cluster_fraction"
        ],
        ascending=[
            False,
            True
        ]
    )
    .head(5)
)


print(
    best_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# FINAL
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
    "FEATURE-SPACE SENSITIVITY COMPLETE"
)

print(
    "=" * 60
)