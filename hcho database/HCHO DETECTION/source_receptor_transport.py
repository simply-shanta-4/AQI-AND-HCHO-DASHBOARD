import os
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = "features/HCHO_FIRE_WIND_2025.csv"

CLUSTER_FILE = "features/HCHO_SOURCE_CLUSTERS_HDBSCAN_2025.csv"

WIND_REGION_FILE = (
    "features/HCHO_SOURCE_REGIONS_WIND_VALIDATED_2025.csv"
)

OUTPUT_MONTHLY = (
    "features/SOURCE_RECEPTOR_TRANSPORT_MONTHLY_2025.csv"
)

OUTPUT_REGION = (
    "features/SOURCE_RECEPTOR_TRANSPORT_REGIONS_2025.csv"
)

OUTPUT_MAP_DATA = (
    "features/SOURCE_RECEPTOR_TRANSPORT_MAP_2025.csv"
)


# ============================================================
# TRANSPORT PARAMETERS
# ============================================================

# Ignore the immediate source area.
MIN_DISTANCE_KM = 25.0

# Maximum transport distance considered.
MAX_DISTANCE_KM = 300.0

# Downwind sector half-angle.
# ±45 degrees around the transport direction.
DOWNWIND_HALF_ANGLE = 45.0

# Upwind sector is the opposite ±45 degrees.
UPWIND_HALF_ANGLE = 45.0


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 60)
print("SOURCE → RECEPTOR TRANSPORT ANALYSIS")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading feature dataset...")

features = pd.read_csv(
    FEATURE_FILE
)

print(
    "Feature rows:",
    len(features)
)


print("\nLoading HDBSCAN clusters...")

clusters = pd.read_csv(
    CLUSTER_FILE
)

print(
    "Cluster rows:",
    len(clusters)
)


print("\nLoading wind-validated regions...")

regions = pd.read_csv(
    WIND_REGION_FILE
)

print(
    "Source regions:",
    len(regions)
)


# ============================================================
# REQUIRED COLUMNS
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
    "wind_speed"
]

required_clusters = [
    "latitude",
    "longitude",
    "cluster_id"
]

required_regions = [
    "cluster_id",
    "mean_latitude",
    "mean_longitude"
]


for col in required_features:

    if col not in features.columns:

        raise ValueError(
            f"Missing feature column: {col}"
        )


for col in required_clusters:

    if col not in clusters.columns:

        raise ValueError(
            f"Missing cluster column: {col}"
        )


for col in required_regions:

    if col not in regions.columns:

        raise ValueError(
            f"Missing region column: {col}"
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

regions = regions.replace(
    [np.inf, -np.inf],
    np.nan
)


features = features.dropna(
    subset=[
        "month",
        "latitude",
        "longitude",
        "HCHO",
        "u10",
        "v10",
        "wind_speed"
    ]
).copy()


clusters = clusters.dropna(
    subset=[
        "latitude",
        "longitude",
        "cluster_id"
    ]
).copy()


regions = regions.dropna(
    subset=[
        "cluster_id",
        "mean_latitude",
        "mean_longitude"
    ]
).copy()


# ============================================================
# REMOVE HDBSCAN NOISE
# ============================================================

clusters = clusters[
    clusters["cluster_id"] != -1
].copy()


# ============================================================
# SOURCE CELL → REGION MAPPING
# ============================================================

source_cells = clusters[
    [
        "latitude",
        "longitude",
        "cluster_id"
    ]
].drop_duplicates()


print(
    "\nNon-noise source cells:",
    len(source_cells)
)


print(
    "HDBSCAN source regions:",
    source_cells[
        "cluster_id"
    ].nunique()
)


# ============================================================
# UNIQUE HCHO GRID
# ============================================================

print(
    "\nPreparing HCHO receptor grid..."
)


grid = features[
    [
        "latitude",
        "longitude"
    ]
].drop_duplicates().reset_index(
    drop=True
)


print(
    "Unique receptor cells:",
    len(grid)
)


grid_lat = grid[
    "latitude"
].to_numpy(
    dtype=float
)

grid_lon = grid[
    "longitude"
].to_numpy(
    dtype=float
)


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * R
        * np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# BEARING
# ============================================================
#
# Bearing FROM source TO receptor.
#
# 0   = north
# 90  = east
# 180 = south
# 270 = west
#
# ============================================================

def bearing(
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlon = np.radians(
        lon2 - lon1
    )

    y = (
        np.sin(dlon)
        * np.cos(lat2)
    )

    x = (
        np.cos(lat1)
        * np.sin(lat2)
        -
        np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    angle = np.degrees(
        np.arctan2(
            y,
            x
        )
    )

    return (
        angle + 360.0
    ) % 360.0


# ============================================================
# ANGULAR DIFFERENCE
# ============================================================

def angular_difference(
    a,
    b
):

    return np.abs(
        (
            a - b + 180.0
        ) % 360.0
        - 180.0
    )


# ============================================================
# BUILD MONTHLY RECEPTOR ARRAYS
# ============================================================

print(
    "\nPreparing monthly receptor data..."
)


monthly_receptors = {}


for month in sorted(
    features["month"].unique()
):

    month_data = features[
        features["month"] == month
    ][
        [
            "latitude",
            "longitude",
            "HCHO"
        ]
    ].copy()


    # Multiple rows should not exist for the same
    # latitude/longitude/month, but aggregate defensively.

    month_data = (
        month_data
        .groupby(
            [
                "latitude",
                "longitude"
            ],
            as_index=False
        )
        ["HCHO"]
        .mean()
    )


    monthly_receptors[
        int(month)
    ] = month_data


    print(
        f"Month {int(month):02d}: "
        f"{len(month_data)} receptor cells"
    )


# ============================================================
# SOURCE MONTHLY WIND
# ============================================================

print(
    "\nCalculating monthly wind vectors for source regions..."
)


source_observations = features.merge(
    source_cells,
    on=[
        "latitude",
        "longitude"
    ],
    how="inner"
)


print(
    "Source-region monthly observations:",
    len(source_observations)
)


# ============================================================
# MONTHLY SOURCE WIND
# ============================================================

source_wind = (
    source_observations
    .groupby(
        [
            "cluster_id",
            "month"
        ],
        as_index=False
    )
    .agg(
        mean_u10=(
            "u10",
            "mean"
        ),
        mean_v10=(
            "v10",
            "mean"
        ),
        mean_wind_speed=(
            "wind_speed",
            "mean"
        ),
        source_HCHO=(
            "HCHO",
            "mean"
        ),
        source_fire_count=(
            "fire_count",
            "sum"
        ),
        source_FRP=(
            "FRP",
            "sum"
        )
    )
)


# ============================================================
# SOURCE WIND DIRECTION
# ============================================================
#
# u10/v10 are wind components.
#
# atan2(u, v) gives direction TOWARD which the air moves.
#
# ============================================================

source_wind[
    "transport_direction"
] = (

    np.degrees(
        np.arctan2(
            source_wind[
                "mean_u10"
            ],
            source_wind[
                "mean_v10"
            ]
        )
    )

    + 360.0

) % 360.0


# ============================================================
# RESULT STORAGE
# ============================================================

monthly_results = []

map_results = []


# ============================================================
# SOURCE → RECEPTOR ANALYSIS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "CALCULATING SOURCE → RECEPTOR TRANSPORT"
)

print(
    "=" * 60
)


cluster_ids = sorted(
    source_cells[
        "cluster_id"
    ].unique()
)


for counter, cluster_id in enumerate(
    cluster_ids,
    start=1
):

    region = regions[
        regions["cluster_id"]
        == cluster_id
    ]


    if len(region) == 0:

        continue


    source_lat = float(
        region.iloc[0][
            "mean_latitude"
        ]
    )

    source_lon = float(
        region.iloc[0][
            "mean_longitude"
        ]
    )


    print(
        f"\nRegion {counter}/"
        f"{len(cluster_ids)} "
        f"(Cluster {cluster_id})"
    )

    print(
        f"Source centroid: "
        f"{source_lat:.4f}, "
        f"{source_lon:.4f}"
    )


    # --------------------------------------------------------
    # SOURCE MONTHS
    # --------------------------------------------------------

    source_months = source_wind[
        source_wind["cluster_id"]
        == cluster_id
    ]


    for _, wind_row in (
        source_months.iterrows()
    ):

        month = int(
            wind_row["month"]
        )


        if month not in monthly_receptors:

            continue


        receptors = (
            monthly_receptors[
                month
            ]
        )


        receptor_lat = receptors[
            "latitude"
        ].to_numpy(
            dtype=float
        )

        receptor_lon = receptors[
            "longitude"
        ].to_numpy(
            dtype=float
        )

        receptor_hcho = receptors[
            "HCHO"
        ].to_numpy(
            dtype=float
        )


        # ----------------------------------------------------
        # DISTANCE
        # ----------------------------------------------------

        distance = haversine_distance(
            source_lat,
            source_lon,
            receptor_lat,
            receptor_lon
        )


        # ----------------------------------------------------
        # BEARING
        # ----------------------------------------------------

        receptor_bearing = bearing(
            source_lat,
            source_lon,
            receptor_lat,
            receptor_lon
        )


        transport_direction = float(
            wind_row[
                "transport_direction"
            ]
        )


        angle_difference = angular_difference(
            receptor_bearing,
            transport_direction
        )


        # ----------------------------------------------------
        # DOWNWIND
        # ----------------------------------------------------

        downwind_mask = (

            (distance >= MIN_DISTANCE_KM)

            &

            (distance <= MAX_DISTANCE_KM)

            &

            (
                angle_difference
                <= DOWNWIND_HALF_ANGLE
            )
        )


        # ----------------------------------------------------
        # UPWIND
        # ----------------------------------------------------

        opposite_direction = (
            transport_direction
            + 180.0
        ) % 360.0


        upwind_difference = (
            angular_difference(
                receptor_bearing,
                opposite_direction
            )
        )


        upwind_mask = (

            (distance >= MIN_DISTANCE_KM)

            &

            (distance <= MAX_DISTANCE_KM)

            &

            (
                upwind_difference
                <= UPWIND_HALF_ANGLE
            )
        )


        # ----------------------------------------------------
        # CROSSWIND
        # ----------------------------------------------------

        crosswind_mask = (

            (distance >= MIN_DISTANCE_KM)

            &

            (distance <= MAX_DISTANCE_KM)

            &

            (~downwind_mask)

            &

            (~upwind_mask)
        )


        # ----------------------------------------------------
        # HCHO VALUES
        # ----------------------------------------------------

        downwind_hcho = receptor_hcho[
            downwind_mask
        ]

        upwind_hcho = receptor_hcho[
            upwind_mask
        ]

        crosswind_hcho = receptor_hcho[
            crosswind_mask
        ]


        # ----------------------------------------------------
        # MEANS
        # ----------------------------------------------------

        if len(downwind_hcho) > 0:

            mean_downwind = float(
                np.mean(
                    downwind_hcho
                )
            )

        else:

            mean_downwind = np.nan


        if len(upwind_hcho) > 0:

            mean_upwind = float(
                np.mean(
                    upwind_hcho
                )
            )

        else:

            mean_upwind = np.nan


        if len(crosswind_hcho) > 0:

            mean_crosswind = float(
                np.mean(
                    crosswind_hcho
                )
            )

        else:

            mean_crosswind = np.nan


        # ----------------------------------------------------
        # ENHANCEMENT
        # ----------------------------------------------------

        if (
            np.isfinite(
                mean_downwind
            )
            and
            np.isfinite(
                mean_upwind
            )
        ):

            absolute_enhancement = (
                mean_downwind
                - mean_upwind
            )


            if mean_upwind != 0:

                relative_enhancement = (
                    (
                        mean_downwind
                        -
                        mean_upwind
                    )
                    /
                    abs(mean_upwind)
                )

            else:

                relative_enhancement = np.nan

        else:

            absolute_enhancement = np.nan

            relative_enhancement = np.nan


        # ----------------------------------------------------
        # DOWNWIND / UPWIND RATIO
        # ----------------------------------------------------

        if (
            np.isfinite(
                mean_downwind
            )
            and
            np.isfinite(
                mean_upwind
            )
            and
            mean_upwind > 0
        ):

            downwind_upwind_ratio = (
                mean_downwind
                /
                mean_upwind
            )

        else:

            downwind_upwind_ratio = np.nan


        # ----------------------------------------------------
        # STORE MONTHLY RESULT
        # ----------------------------------------------------

        monthly_results.append({

            "cluster_id":
                int(cluster_id),

            "month":
                month,

            "source_latitude":
                source_lat,

            "source_longitude":
                source_lon,

            "source_HCHO":
                wind_row[
                    "source_HCHO"
                ],

            "source_fire_count":
                wind_row[
                    "source_fire_count"
                ],

            "source_FRP":
                wind_row[
                    "source_FRP"
                ],

            "mean_wind_speed":
                wind_row[
                    "mean_wind_speed"
                ],

            "transport_direction":
                transport_direction,

            "downwind_cells":
                int(
                    np.sum(
                        downwind_mask
                    )
                ),

            "upwind_cells":
                int(
                    np.sum(
                        upwind_mask
                    )
                ),

            "crosswind_cells":
                int(
                    np.sum(
                        crosswind_mask
                    )
                ),

            "mean_downwind_HCHO":
                mean_downwind,

            "mean_upwind_HCHO":
                mean_upwind,

            "mean_crosswind_HCHO":
                mean_crosswind,

            "absolute_HCHO_enhancement":
                absolute_enhancement,

            "relative_HCHO_enhancement":
                relative_enhancement,

            "downwind_upwind_ratio":
                downwind_upwind_ratio
        })


        # ----------------------------------------------------
        # MAP DATA
        # ----------------------------------------------------

        # Save only downwind cells for mapping.

        selected = downwind_mask

        if np.any(selected):

            map_frame = pd.DataFrame({

                "cluster_id":
                    int(cluster_id),

                "month":
                    month,

                "source_latitude":
                    source_lat,

                "source_longitude":
                    source_lon,

                "receptor_latitude":
                    receptor_lat[selected],

                "receptor_longitude":
                    receptor_lon[selected],

                "distance_km":
                    distance[selected],

                "bearing":
                    receptor_bearing[selected],

                "transport_direction":
                    transport_direction,

                "HCHO":
                    receptor_hcho[selected],

                "angle_difference":
                    angle_difference[selected]
            })


            map_results.append(
                map_frame
            )


# ============================================================
# DATAFRAME
# ============================================================

monthly_results = pd.DataFrame(
    monthly_results
)


# ============================================================
# MONTHLY OUTPUT
# ============================================================

monthly_results.to_csv(
    OUTPUT_MONTHLY,
    index=False
)


# ============================================================
# REGION-LEVEL SUMMARY
# ============================================================

print(
    "\nCreating regional transport summary..."
)


region_summary = (
    monthly_results
    .groupby(
        "cluster_id",
        as_index=False
    )
    .agg(

        months_analyzed=(
            "month",
            "nunique"
        ),

        mean_downwind_HCHO=(
            "mean_downwind_HCHO",
            "mean"
        ),

        mean_upwind_HCHO=(
            "mean_upwind_HCHO",
            "mean"
        ),

        mean_crosswind_HCHO=(
            "mean_crosswind_HCHO",
            "mean"
        ),

        mean_absolute_enhancement=(
            "absolute_HCHO_enhancement",
            "mean"
        ),

        mean_relative_enhancement=(
            "relative_HCHO_enhancement",
            "mean"
        ),

        mean_downwind_upwind_ratio=(
            "downwind_upwind_ratio",
            "mean"
        ),

        median_downwind_upwind_ratio=(
            "downwind_upwind_ratio",
            "median"
        ),

        total_downwind_cells=(
            "downwind_cells",
            "sum"
        ),

        total_upwind_cells=(
            "upwind_cells",
            "sum"
        ),

        mean_source_fire_count=(
            "source_fire_count",
            "mean"
        ),

        mean_source_FRP=(
            "source_FRP",
            "mean"
        ),

        mean_wind_speed=(
            "mean_wind_speed",
            "mean"
        )
    )
)


# ============================================================
# CONSISTENCY OF POSITIVE TRANSPORT SIGNAL
# ============================================================

monthly_results[
    "positive_transport_signal"
] = (
    monthly_results[
        "absolute_HCHO_enhancement"
    ] > 0
)


positive_signal = (
    monthly_results
    .groupby(
        "cluster_id"
    )[
        "positive_transport_signal"
    ]
    .mean()
    .reset_index(
        name="positive_transport_month_fraction"
    )
)


region_summary = region_summary.merge(
    positive_signal,
    on="cluster_id",
    how="left"
)


# ============================================================
# MERGE ORIGINAL REGION INFORMATION
# ============================================================

region_summary = region_summary.merge(
    regions,
    on="cluster_id",
    how="left"
)


# ============================================================
# TRANSPORT EVIDENCE SCORE
# ============================================================
#
# This is a diagnostic score, NOT a probability.
#
# Components:
#
# 1. Downwind/upwind ratio
# 2. Positive enhancement frequency
# 3. Absolute enhancement
#
# ============================================================

def minmax_safe(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid = series[
        np.isfinite(series)
    ]

    if len(valid) == 0:

        return pd.Series(
            0.0,
            index=series.index
        )


    low = valid.min()
    high = valid.max()


    if high == low:

        return pd.Series(
            0.5,
            index=series.index
        )


    result = (
        series - low
    ) / (
        high - low
    )


    return result.fillna(0.0)


ratio_score = minmax_safe(
    region_summary[
        "mean_downwind_upwind_ratio"
    ]
)


enhancement_score = minmax_safe(
    region_summary[
        "mean_absolute_enhancement"
    ]
)


consistency_score = (
    region_summary[
        "positive_transport_month_fraction"
    ]
    .fillna(0.0)
)


region_summary[
    "transport_evidence_score"
] = (

    0.40
    * ratio_score

    +

    0.35
    * enhancement_score

    +

    0.25
    * consistency_score
)


# ============================================================
# FINAL TRANSPORT RANK
# ============================================================

region_summary = (
    region_summary
    .sort_values(
        "transport_evidence_score",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


region_summary[
    "transport_rank"
] = (
    region_summary.index + 1
)


# ============================================================
# SAVE REGION SUMMARY
# ============================================================

region_summary.to_csv(
    OUTPUT_REGION,
    index=False
)


# ============================================================
# MAP DATA
# ============================================================

if len(map_results) > 0:

    map_data = pd.concat(
        map_results,
        ignore_index=True
    )

else:

    map_data = pd.DataFrame()


map_data.to_csv(
    OUTPUT_MAP_DATA,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "SOURCE → RECEPTOR TRANSPORT RESULTS"
)

print(
    "=" * 60
)


display_columns = [

    "transport_rank",

    "cluster_id",

    "cells",

    "mean_latitude",

    "mean_longitude",

    "months_analyzed",

    "mean_downwind_HCHO",

    "mean_upwind_HCHO",

    "mean_crosswind_HCHO",

    "mean_absolute_enhancement",

    "mean_relative_enhancement",

    "mean_downwind_upwind_ratio",

    "positive_transport_month_fraction",

    "mean_source_fire_count",

    "mean_source_FRP",

    "transport_evidence_score"
]


print(
    region_summary[
        display_columns
    ].head(20).to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ============================================================
# SAVE MESSAGE
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FILES SAVED"
)

print(
    "=" * 60
)

print(
    OUTPUT_MONTHLY
)

print(
    OUTPUT_REGION
)

print(
    OUTPUT_MAP_DATA
)


print(
    "\n" + "=" * 60
)

print(
    "SOURCE → RECEPTOR TRANSPORT COMPLETE"
)

print(
    "=" * 60
)