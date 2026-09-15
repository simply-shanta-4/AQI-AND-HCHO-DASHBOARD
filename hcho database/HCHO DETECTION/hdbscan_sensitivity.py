import os
import numpy as np
import pandas as pd
import hdbscan


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "features/SPATIAL_SOURCE_CANDIDATES_2025.csv"
)

OUTPUT_FILE = (
    "features/HDBSCAN_SENSITIVITY_2025.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("HDBSCAN SENSITIVITY ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(
    "Spatial candidates:",
    len(df)
)


# ============================================================
# VALIDATE COORDINATES
# ============================================================

df = df.dropna(
    subset=[
        "latitude",
        "longitude"
    ]
).copy()

print(
    "Valid spatial candidates:",
    len(df)
)


# ============================================================
# CONVERT LAT/LON TO APPROXIMATE KM
# ============================================================
#
# HDBSCAN should operate on distance in approximately
# physical units rather than raw degrees.
#
# Latitude:
#   1 degree ≈ 111.32 km
#
# Longitude:
#   distance depends on latitude
#
# We use a local latitude correction for longitude.
# ============================================================

latitude = df[
    "latitude"
].to_numpy(
    dtype=float
)

longitude = df[
    "longitude"
].to_numpy(
    dtype=float
)


latitude_km = (
    latitude * 111.32
)

longitude_km = (
    longitude
    * 111.32
    * np.cos(
        np.radians(
            latitude
        )
    )
)


X = np.column_stack(
    [
        longitude_km,
        latitude_km
    ]
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
# RUN SENSITIVITY ANALYSIS
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

            cluster_selection_method=(
                "eom"
            )
        )


        labels = clusterer.fit_predict(
            X
        )


        # ----------------------------------------------------
        # CLUSTER STATISTICS
        # ----------------------------------------------------

        unique_labels = set(
            labels
        )

        cluster_labels = [
            label
            for label in unique_labels
            if label != -1
        ]


        n_clusters = len(
            cluster_labels
        )


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

            median_cluster_size = (
                float(
                    np.median(
                        cluster_sizes
                    )
                )
            )

        else:

            largest_cluster = 0

            smallest_cluster = 0

            median_cluster_size = 0


        # ----------------------------------------------------
        # HDBSCAN PERSISTENCE
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
# RESULTS DATAFRAME
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
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "HDBSCAN SENSITIVITY RESULTS"
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
# BEST PERSISTENCE
# ============================================================

if len(results_df) > 0:

    best = results_df.sort_values(
        [
            "mean_persistence",
            "max_persistence"
        ],
        ascending=False
    ).iloc[0]


    print(
        "\n========== HIGHEST PERSISTENCE =========="
    )

    print(
        "min_cluster_size:",
        int(
            best[
                "min_cluster_size"
            ]
        )
    )

    print(
        "min_samples:",
        int(
            best[
                "min_samples"
            ]
        )
    )

    print(
        "clusters:",
        int(
            best[
                "clusters"
            ]
        )
    )

    print(
        "noise:",
        f"{best['noise_percent']:.2f}%"
    )

    print(
        "mean persistence:",
        f"{best['mean_persistence']:.3f}"
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
    "HDBSCAN SENSITIVITY ANALYSIS COMPLETE"
)

print(
    "=" * 60
)