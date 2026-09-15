import os
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "features/HCHO_FIRE_WIND_2025.csv"
)

OUTPUT_FILE = (
    "features/HCHO_SOURCE_CANDIDATES_2025.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("CREATING SOURCE CANDIDATE DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(
    "Total feature rows:",
    len(df)
)


# ============================================================
# SELECT HCHO HOTSPOTS
# ============================================================

hotspots = df[
    df["HCHO_hotspot"] == True
].copy()

print(
    "HCHO hotspots:",
    len(hotspots)
)


# ============================================================
# SELECT FIRE-SUPPORTED HOTSPOTS
# ============================================================

candidates = hotspots[
    hotspots["fire_count"] > 0
].copy()

print(
    "Fire-supported HCHO hotspots:",
    len(candidates)
)


# ============================================================
# CHECK
# ============================================================

if len(candidates) == 0:

    raise ValueError(
        "No fire-supported HCHO hotspots found."
    )


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def minmax(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


# ============================================================
# NORMALIZE PHYSICAL VARIABLES
# ============================================================

candidates["HCHO_norm"] = minmax(
    candidates["HCHO"]
)

candidates["fire_norm"] = minmax(
    candidates["fire_count"]
)

candidates["FRP_norm"] = minmax(
    candidates["FRP"]
)


# ============================================================
# SOURCE SCORE
# ============================================================
#
# The score combines:
#
# HCHO enhancement       → 40%
# Fire activity          → 30%
# FRP                    → 30%
#
# Wind is retained for
# later transport analysis
# rather than being treated
# as direct evidence of
# source strength.
#
# ============================================================

candidates["source_score"] = (

    0.40 *
    candidates["HCHO_norm"]

    +

    0.30 *
    candidates["fire_norm"]

    +

    0.30 *
    candidates["FRP_norm"]

)


# ============================================================
# WIND INFORMATION
# ============================================================

# Wind speed is retained.
#
# We do NOT multiply source_score by wind speed because
# strong wind does not necessarily mean stronger emissions.
#
# Wind will be used later to investigate transport direction.

candidates["wind_speed"] = (
    pd.to_numeric(
        candidates["wind_speed"],
        errors="coerce"
    )
)

candidates["wind_direction"] = (
    pd.to_numeric(
        candidates["wind_direction"],
        errors="coerce"
    )
)


# ============================================================
# SORT BY SOURCE SCORE
# ============================================================

candidates = candidates.sort_values(
    "source_score",
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

candidates.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SOURCE CANDIDATE DATASET CREATED")
print("=" * 60)

print(
    "Total HCHO hotspots:",
    len(hotspots)
)

print(
    "Fire-supported hotspots:",
    len(candidates)
)

print(
    "HCHO hotspots without fire:",
    len(hotspots) - len(candidates)
)

print(
    "\nSource score range:"
)

print(
    f"{candidates['source_score'].min():.4f}"
    " to "
    f"{candidates['source_score'].max():.4f}"
)

print(
    "\nWind speed range:"
)

print(
    f"{candidates['wind_speed'].min():.3f}"
    " to "
    f"{candidates['wind_speed'].max():.3f}"
)

print(
    "\nWind direction range:"
)

print(
    f"{candidates['wind_direction'].min():.2f}"
    " to "
    f"{candidates['wind_direction'].max():.2f}"
)


# ============================================================
# TOP 20
# ============================================================

print(
    "\n========== TOP 20 SOURCE CANDIDATES =========="
)

display_columns = [
    "month",
    "latitude",
    "longitude",
    "HCHO",
    "fire_count",
    "FRP",
    "wind_speed",
    "wind_direction",
    "source_score"
]

print(
    candidates[
        display_columns
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
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
    "SOURCE CANDIDATE CREATION COMPLETE"
)

print(
    "=" * 60
)