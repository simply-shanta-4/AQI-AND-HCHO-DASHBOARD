import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "features/HCHO_SOURCE_CANDIDATES_2025.csv"
)

OUTPUT_FILE = (
    "features/SPATIAL_SOURCE_CANDIDATES_2025.csv"
)


# ============================================================
# LOAD
# ============================================================

print("\n" + "=" * 60)
print("CREATING SPATIAL SOURCE CANDIDATES")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(
    "Monthly source candidates:",
    len(df)
)


# ============================================================
# ROUND COORDINATES TO STABLE HCHO GRID PRECISION
# ============================================================
#
# HCHO pixel centres are approximately:
# latitude  ≈ 0.0875 degrees
# longitude ≈ 0.101 degrees
#
# We therefore identify repeated spatial observations
# using the actual HCHO pixel coordinates rounded to
# 5 decimal places.
#
# This preserves the HCHO grid rather than creating
# an arbitrary coarse spatial grid.
# ============================================================

df["latitude_key"] = (
    df["latitude"].round(5)
)

df["longitude_key"] = (
    df["longitude"].round(5)
)


# ============================================================
# AGGREGATE TEMPORALLY
# ============================================================

spatial = (
    df
    .groupby(
        [
            "latitude_key",
            "longitude_key"
        ],
        as_index=False
    )
    .agg(

        # Representative location
        latitude=(
            "latitude",
            "mean"
        ),

        longitude=(
            "longitude",
            "mean"
        ),

        # Temporal persistence
        active_months=(
            "month",
            "nunique"
        ),

        # HCHO
        mean_HCHO=(
            "HCHO",
            "mean"
        ),

        max_HCHO=(
            "HCHO",
            "max"
        ),

        # Fire activity
        total_fire_count=(
            "fire_count",
            "sum"
        ),

        max_fire_count=(
            "fire_count",
            "max"
        ),

        # FRP
        total_FRP=(
            "FRP",
            "sum"
        ),

        max_FRP=(
            "FRP",
            "max"
        ),

        # Source score
        mean_source_score=(
            "source_score",
            "mean"
        ),

        max_source_score=(
            "source_score",
            "max"
        ),

        # Wind
        mean_wind_speed=(
            "wind_speed",
            "mean"
        )
    )
)


# ============================================================
# REMOVE TEMPORARY KEYS
# ============================================================

spatial = spatial.drop(
    columns=[
        "latitude_key",
        "longitude_key"
    ]
)


# ============================================================
# SORT
# ============================================================

spatial = spatial.sort_values(
    [
        "mean_source_score",
        "active_months"
    ],
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "features",
    exist_ok=True
)

spatial.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SPATIAL SOURCE CANDIDATES CREATED")
print("=" * 60)

print(
    "Monthly candidates:",
    len(df)
)

print(
    "Unique spatial candidates:",
    len(spatial)
)

print(
    "Maximum active months:",
    spatial["active_months"].max()
)

print(
    "Mean active months:",
    f"{spatial['active_months'].mean():.3f}"
)


# ============================================================
# ACTIVE-MONTH DISTRIBUTION
# ============================================================

print(
    "\n========== ACTIVE MONTH DISTRIBUTION =========="
)

print(
    spatial[
        "active_months"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# TOP SPATIAL CANDIDATES
# ============================================================

print(
    "\n========== TOP 20 SPATIAL CANDIDATES =========="
)

display_columns = [
    "latitude",
    "longitude",
    "active_months",
    "mean_HCHO",
    "max_HCHO",
    "total_fire_count",
    "total_FRP",
    "mean_source_score",
    "max_source_score"
]

print(
    spatial[
        display_columns
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# OUTPUT
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
    "SPATIAL CANDIDATE CREATION COMPLETE"
)

print(
    "=" * 60
)