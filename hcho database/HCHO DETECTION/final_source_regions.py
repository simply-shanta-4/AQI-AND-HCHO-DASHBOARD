import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from location_metadata import CLUSTER_LOCATIONS, get_location_name, INDIAN_CLUSTER_IDS
except ImportError:
    from location_metadata import CLUSTER_LOCATIONS, get_location_name, INDIAN_CLUSTER_IDS


# ============================================================
# FINAL HDBSCAN SOURCE-REGION RANKING
# ============================================================

print("\n" + "=" * 70)
print("FINAL HDBSCAN SOURCE-REGION ANALYSIS (DOMESTIC INDIA ONLY)")
print("=" * 70)


# ============================================================
# FILES
# ============================================================

WIND_FILE = (
    "features/HCHO_SOURCE_REGIONS_WIND_VALIDATED_2025.csv"
)

SIG_FILE = (
    "features/SOURCE_RECEPTOR_TRANSPORT_SIGNIFICANCE_2025.csv"
)

OUTPUT_CSV = (
    "features/FINAL_HDBSCAN_SOURCE_REGIONS_2025.csv"
)

OUTPUT_MAP = (
    "features/FINAL_HDBSCAN_SOURCE_REGIONS_2025.png"
)


# ============================================================
# LOAD WIND-VALIDATED HDBSCAN REGIONS
# ============================================================

print("\nLoading HDBSCAN wind-validated regions...")

wind = pd.read_csv(
    WIND_FILE
)

wind["cluster_id"] = wind["cluster_id"].astype(int)
wind = wind[wind["cluster_id"].isin(INDIAN_CLUSTER_IDS)].copy().reset_index(drop=True)

print(
    "Domestic Indian Wind-validated regions:",
    len(wind)
)



# ============================================================
# LOAD STATISTICAL RESULTS
# ============================================================

print(
    "\nLoading transport significance results..."
)

sig = pd.read_csv(
    SIG_FILE
)

print(
    "Statistical regions:",
    len(sig)
)


# ============================================================
# CHECK CLUSTER ID
# ============================================================

if "cluster_id" not in wind.columns:

    raise ValueError(
        "cluster_id missing from wind-validation file"
    )


if "cluster_id" not in sig.columns:

    raise ValueError(
        "cluster_id missing from significance file"
    )


# ============================================================
# MERGE
# ============================================================

print(
    "\nMerging HDBSCAN + wind + statistical evidence..."
)


df = pd.merge(
    wind,
    sig,
    on="cluster_id",
    how="left",
    suffixes=(
        "",
        "_stat"
    )
)


print(
    "Merged regions:",
    len(df)
)


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

numeric_columns = [
    "mean_source_score",
    "total_fire_count",
    "total_FRP",
    "cluster_persistence",
    "mean_wind_speed",
    "resultant_wind_speed",
    "directional_consistency",
    "wind_transport_score",
    "wind_validated_score",
    "observed_mean_enhancement",
    "observed_downwind_upwind_ratio",
    "effect_size",
    "p_value_downwind_positive",
    "fdr_adjusted_p"
]


for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================

def minmax(series):

    series = series.astype(float)

    minimum = series.min()

    maximum = series.max()

    if (
        pd.isna(minimum)
        or
        pd.isna(maximum)
        or
        maximum == minimum
    ):

        return pd.Series(
            1.0,
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# SOURCE SCORE
# ============================================================

df[
    "source_score_component"
] = minmax(
    df["mean_source_score"]
)


# ============================================================
# FIRE ACTIVITY
# ============================================================

df[
    "fire_component"
] = minmax(
    np.log1p(
        df["total_fire_count"]
    )
)


# ============================================================
# FRP ACTIVITY
# ============================================================

df[
    "frp_component"
] = minmax(
    np.log1p(
        df["total_FRP"]
    )
)


# ============================================================
# HDBSCAN PERSISTENCE
# ============================================================

df[
    "persistence_component"
] = minmax(
    df["cluster_persistence"]
)


# ============================================================
# WIND TRANSPORT
# ============================================================

df[
    "wind_component"
] = minmax(
    df["wind_transport_score"]
)


# ============================================================
# DOWNWIND HCHO ENHANCEMENT
# ============================================================

df[
    "enhancement_component"
] = minmax(
    df[
        "observed_mean_enhancement"
    ]
)


# ============================================================
# EFFECT SIZE
# ============================================================

df[
    "effect_component"
] = minmax(
    np.maximum(
        df["effect_size"],
        0
    )
)


# ============================================================
# STATISTICAL EVIDENCE
# ============================================================
#
# Smaller FDR-adjusted p-value = stronger evidence.
#
# We transform q-values into an evidence score.
#
# q >= 0.05 -> 0
# q < 0.05  -> increasing evidence
#
# ============================================================

q = df[
    "fdr_adjusted_p"
].astype(float)


df[
    "statistical_evidence_component"
] = np.where(

    q < 0.05,

    np.clip(
        -np.log10(
            np.maximum(
                q,
                1e-10
            )
        )
        / 3.0,
        0,
        1
    ),

    0.0
)


# ============================================================
# FINAL EVIDENCE SCORE
# ============================================================
#
# The score deliberately gives substantial weight to:
#
# 1. wind transport
# 2. statistical downwind enhancement
# 3. source evidence
# 4. HDBSCAN persistence
#
# Fire/FRP are supporting evidence.
#
# ============================================================

df[
    "final_source_score"
] = (

    0.20
    * df[
        "source_score_component"
    ]

    +

    0.10
    * df[
        "fire_component"
    ]

    +

    0.10
    * df[
        "frp_component"
    ]

    +

    0.15
    * df[
        "persistence_component"
    ]

    +

    0.15
    * df[
        "wind_component"
    ]

    +

    0.10
    * df[
        "enhancement_component"
    ]

    +

    0.10
    * df[
        "effect_component"
    ]

    +

    0.10
    * df[
        "statistical_evidence_component"
    ]
)


# ============================================================
# STATISTICAL CLASS
# ============================================================

df[
    "statistical_class"
] = "Not significant"


df.loc[
    df[
        "fdr_adjusted_p"
    ] < 0.05,
    "statistical_class"
] = "FDR q < 0.05"


df.loc[
    df[
        "fdr_adjusted_p"
    ] < 0.01,
    "statistical_class"
] = "FDR q < 0.01"


# ============================================================
# SOURCE CONFIDENCE CLASS
# ============================================================

df[
    "source_confidence"
] = "Low"


df.loc[
    (
        (df["fdr_adjusted_p"] < 0.05)
        &
        (df["wind_transport_score"] >=
         df["wind_transport_score"].median())
    ),
    "source_confidence"
] = "Moderate"


df.loc[
    (
        (df["fdr_adjusted_p"] < 0.05)
        &
        (df["wind_transport_score"] >=
         df["wind_transport_score"].median())
        &
        (df["effect_size"] >= 1.0)
        &
        (df["cluster_persistence"] >=
         df["cluster_persistence"].median())
    ),
    "source_confidence"
] = "High"


# ============================================================
# FINAL RANK
# ============================================================

df = df.sort_values(
    [
        "final_source_score",
        "fdr_adjusted_p",
        "effect_size"
    ],
    ascending=[
        False,
        True,
        False
    ]
).reset_index(
    drop=True
)


df[
    "final_source_rank"
] = (
    df.index + 1
)


# ============================================================
# SAVE CSV
# ============================================================

os.makedirs(
    "features",
    exist_ok=True
)


df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL HDBSCAN SOURCE-REGION RANKING"
)

print(
    "=" * 70
)


summary_columns = [
    "final_source_rank",
    "cluster_id",
    "cells",
    "mean_latitude",
    "mean_longitude",
    "mean_source_score",
    "total_fire_count",
    "total_FRP",
    "cluster_persistence",
    "wind_transport_score",
    "observed_mean_enhancement",
    "observed_downwind_upwind_ratio",
    "effect_size",
    "fdr_adjusted_p",
    "statistical_class",
    "source_confidence",
    "final_source_score"
]


summary_columns = [
    c
    for c in summary_columns
    if c in df.columns
]


print(
    df[
        summary_columns
    ].head(20).to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL EVIDENCE SUMMARY"
)

print(
    "=" * 70
)


print(
    "Total HDBSCAN regions:",
    len(df)
)


print(
    "FDR q < 0.05:",
    (
        df[
            "fdr_adjusted_p"
        ] < 0.05
    ).sum()
)


print(
    "FDR q < 0.01:",
    (
        df[
            "fdr_adjusted_p"
        ] < 0.01
    ).sum()
)


print(
    "High-confidence regions:",
    (
        df[
            "source_confidence"
        ] == "High"
    ).sum()
)


print(
    "Moderate-confidence regions:",
    (
        df[
            "source_confidence"
        ] == "Moderate"
    ).sum()
)


# ============================================================
# CREATE MAP
# ============================================================

print(
    "\nCreating final HDBSCAN source-region map..."
)


fig, ax = plt.subplots(
    figsize=(14, 10)
)


# ------------------------------------------------------------
# Plot all regions
# ------------------------------------------------------------

scatter = ax.scatter(

    df["mean_longitude"],

    df["mean_latitude"],

    s=(
        40
        +
        260
        *
        df[
            "final_source_score"
        ]
    ),

    c=df[
        "final_source_score"
    ],

    cmap="viridis",

    alpha=0.85,

    edgecolors="black",

    linewidths=0.5
)


# ============================================================
# LABEL TOP 15
# ============================================================

df["location_name"] = df["cluster_id"].apply(lambda cid: get_location_name(cid, "standard"))
df["location_full"] = df["cluster_id"].apply(lambda cid: get_location_name(cid, "full"))
df["location_short"] = df["cluster_id"].apply(lambda cid: get_location_name(cid, "short"))
df["state"] = df["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(int(cid), {}).get("state", "N/A"))
df["zone"] = df["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(int(cid), {}).get("zone", "N/A"))

top_n = min(
    16,
    len(df)
)


for _, row in df.head(
    top_n
).iterrows():

    rank_num = int(row['final_source_rank'])
    loc_lbl = f"#{rank_num} {row['location_short']}"

    ax.annotate(
        loc_lbl,
        (
            row["mean_longitude"],
            row["mean_latitude"]
        ),
        xytext=(
            6,
            6
        ),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.82, edgecolor="#CBD5E1", linewidth=0.5),
        zorder=4
    )


# ============================================================
# AXES
# ============================================================

ax.set_xlabel(
    "Longitude (°E)",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Latitude (°N)",
    fontsize=12,
    fontweight="bold"
)


ax.set_title(
    "Final HDBSCAN Domestic Indian HCHO Source Regions — 2025",
    fontsize=16,
    fontweight="bold",
    pad=14
)
ax.set_xlim(68, 96)
ax.set_ylim(7.5, 36.5)




# ============================================================
# COLORBAR
# ============================================================

cbar = plt.colorbar(
    scatter,
    ax=ax
)

cbar.set_label(
    "Final source evidence score"
)


# ============================================================
# GRID
# ============================================================

ax.grid(
    True,
    alpha=0.3
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()


plt.savefig(
    OUTPUT_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print(
    "\nSaved:"
)

print(
    OUTPUT_CSV
)

print(
    OUTPUT_MAP
)


print(
    "\n" + "=" * 70
)

print(
    "FINAL HDBSCAN SOURCE-REGION ANALYSIS COMPLETE"
)

print(
    "=" * 70
)