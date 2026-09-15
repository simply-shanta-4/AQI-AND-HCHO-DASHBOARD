import os
import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "features/SPATIAL_SOURCE_CANDIDATES_2025.csv"
)

OUTPUT_CSV = (
    "features/HDBSCAN_SOLUTION_COMPARISON_2025.csv"
)

OUTPUT_MAP = (
    "features/HDBSCAN_SOLUTION_COMPARISON_2025.png"
)


# ============================================================
# LOAD
# ============================================================

print("\n" + "=" * 60)
print("HDBSCAN SOLUTION COMPARISON")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(
    "Spatial candidates:",
    len(df)
)


# ============================================================
# COORDINATES → KM
# ============================================================

lat = df["latitude"].to_numpy(
    dtype=float
)

lon = df["longitude"].to_numpy(
    dtype=float
)

lat_km = lat * 111.32

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
# THREE CANDIDATE SOLUTIONS
# ============================================================

solutions = {

    "A_fine":
        {
            "min_cluster_size": 10,
            "min_samples": 5
        },

    "B_intermediate":
        {
            "min_cluster_size": 30,
            "min_samples": 5
        },

    "C_conservative":
        {
            "min_cluster_size": 100,
            "min_samples": 10
        }
}


all_results = []

map_labels = {}


# ============================================================
# RUN EACH SOLUTION
# ============================================================

for name, params in solutions.items():

    print(
        "\n" + "-" * 60
    )

    print(
        name
    )

    print(
        "min_cluster_size:",
        params["min_cluster_size"]
    )

    print(
        "min_samples:",
        params["min_samples"]
    )


    clusterer = hdbscan.HDBSCAN(

        min_cluster_size=(
            params["min_cluster_size"]
        ),

        min_samples=(
            params["min_samples"]
        ),

        metric="euclidean",

        cluster_selection_method="eom"
    )


    labels = clusterer.fit_predict(
        X
    )

    map_labels[name] = labels


    # --------------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------------

    cluster_ids = sorted(
        [
            x
            for x in np.unique(labels)
            if x != -1
        ]
    )

    n_clusters = len(
        cluster_ids
    )

    noise_points = int(
        np.sum(
            labels == -1
        )
    )

    noise_percent = (
        100
        * noise_points
        / len(labels)
    )


    if n_clusters > 0:

        cluster_sizes = [

            int(
                np.sum(
                    labels == cluster_id
                )
            )

            for cluster_id in cluster_ids
        ]

        largest_cluster = max(
            cluster_sizes
        )

        median_cluster = float(
            np.median(
                cluster_sizes
            )
        )

        smallest_cluster = min(
            cluster_sizes
        )

    else:

        largest_cluster = 0
        median_cluster = 0
        smallest_cluster = 0


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

        mean_persistence = 0
        max_persistence = 0


    largest_fraction = (
        largest_cluster
        / len(df)
    )


    # --------------------------------------------------------
    # SUMMARY ROW
    # --------------------------------------------------------

    all_results.append({

        "solution": name,

        "min_cluster_size":
            params["min_cluster_size"],

        "min_samples":
            params["min_samples"],

        "clusters":
            n_clusters,

        "noise_points":
            noise_points,

        "noise_percent":
            noise_percent,

        "largest_cluster":
            largest_cluster,

        "largest_cluster_fraction":
            largest_fraction,

        "median_cluster_size":
            median_cluster,

        "smallest_cluster":
            smallest_cluster,

        "mean_persistence":
            mean_persistence,

        "max_persistence":
            max_persistence
    })


    # --------------------------------------------------------
    # CLUSTER-SPECIFIC PHYSICAL SUMMARY
    # --------------------------------------------------------

    print(
        "\nCluster summary:"
    )

    for cluster_id in cluster_ids:

        subset = df[
            labels == cluster_id
        ]

        print(
            f"\nCluster {cluster_id}"
        )

        print(
            "  cells:",
            len(subset)
        )

        print(
            "  mean latitude:",
            f"{subset['latitude'].mean():.3f}"
        )

        print(
            "  mean longitude:",
            f"{subset['longitude'].mean():.3f}"
        )

        print(
            "  mean HCHO:",
            f"{subset['mean_HCHO'].mean():.6f}"
        )

        print(
            "  mean source score:",
            f"{subset['mean_source_score'].mean():.4f}"
        )

        print(
            "  total fire:",
            int(
                subset[
                    "total_fire_count"
                ].sum()
            )
        )

        print(
            "  total FRP:",
            f"{subset['total_FRP'].sum():.2f}"
        )

        print(
            "  mean active months:",
            f"{subset['active_months'].mean():.2f}"
        )


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison = pd.DataFrame(
    all_results
)

comparison.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "SOLUTION COMPARISON"
)

print(
    "=" * 60
)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# CREATE THREE-PANEL MAP
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 7)
)


for ax, (name, labels) in zip(
    axes,
    map_labels.items()
):

    # --------------------------------------------------------
    # Noise
    # --------------------------------------------------------

    noise_mask = (
        labels == -1
    )

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

            s=3,

            alpha=0.15
        )


    # --------------------------------------------------------
    # Clusters
    # --------------------------------------------------------

    cluster_ids = sorted(
        [
            x
            for x in np.unique(labels)
            if x != -1
        ]
    )


    for cluster_id in cluster_ids:

        mask = (
            labels == cluster_id
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


    ax.set_title(
        name.replace(
            "_",
            " "
        )
    )

    ax.set_xlabel(
        "Longitude (°E)"
    )

    ax.set_ylabel(
        "Latitude (°N)"
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


plt.suptitle(
    "Comparison of Spatial HDBSCAN Solutions — 2025"
)

plt.tight_layout()


plt.savefig(
    OUTPUT_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print(
    "\nSaved comparison:"
)

print(
    OUTPUT_CSV
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
    "HDBSCAN SOLUTION COMPARISON COMPLETE"
)

print(
    "=" * 60
)