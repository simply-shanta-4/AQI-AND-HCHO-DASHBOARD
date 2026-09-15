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
# FINAL HDBSCAN SOURCE-REGION VALIDATION
# ============================================================

INPUT_FILE = (
    "features/FINAL_HDBSCAN_SOURCE_REGIONS_2025.csv"
)

OUTPUT_FILE = (
    "features/FINAL_SOURCE_REGION_VALIDATION_2025.csv"
)

OUTPUT_MAP = (
    "features/FINAL_SOURCE_REGION_VALIDATION_MAP_2025.png"
)

OUTPUT_REPORT = (
    "features/FINAL_SOURCE_REGION_REPORT_2025.csv"
)


print("\n" + "=" * 70)
print("FINAL HDBSCAN SOURCE-REGION VALIDATION (DOMESTIC INDIA ONLY)")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

print("\nLoading final HDBSCAN regions...")

df = pd.read_csv(INPUT_FILE)
df["cluster_id"] = df["cluster_id"].astype(int)
# EXCLUSIVELY KEEP DOMESTIC INDIAN REGIONS
df = df[df["cluster_id"].isin(INDIAN_CLUSTER_IDS)].copy().reset_index(drop=True)

print(
    "Domestic Indian regions:",
    len(df)
)



# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "final_source_rank",
    "cluster_id",
    "cells",
    "mean_latitude",
    "mean_longitude",
    "mean_source_score",
    "total_fire_count",
    "total_FRP",
    "cluster_persistence",
    "mean_wind_speed",
    "resultant_wind_speed",
    "directional_consistency",
    "wind_transport_score",
    "observed_mean_enhancement",
    "observed_downwind_upwind_ratio",
    "effect_size",
    "fdr_adjusted_p",
    "statistical_class",
    "source_confidence",
    "final_source_score"
]


missing = [
    c for c in required
    if c not in df.columns
]


if missing:
    raise ValueError(
        "Missing required columns: "
        + str(missing)
    )


# ============================================================
# CLEAN
# ============================================================

numeric_columns = [
    "final_source_rank",
    "cluster_id",
    "cells",
    "mean_latitude",
    "mean_longitude",
    "mean_source_score",
    "total_fire_count",
    "total_FRP",
    "cluster_persistence",
    "mean_wind_speed",
    "resultant_wind_speed",
    "directional_consistency",
    "wind_transport_score",
    "observed_mean_enhancement",
    "observed_downwind_upwind_ratio",
    "effect_size",
    "fdr_adjusted_p",
    "final_source_score"
]


for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# VALIDATION COMPONENTS
# ============================================================

# ------------------------------------------------------------
# 1. HDBSCAN persistence
# ------------------------------------------------------------

persistence_median = df[
    "cluster_persistence"
].median()


df[
    "hdbscan_valid"
] = (
    df[
        "cluster_persistence"
    ]
    >= persistence_median
)


# ------------------------------------------------------------
# 2. Source evidence
# ------------------------------------------------------------

source_score_median = df[
    "mean_source_score"
].median()


df[
    "source_evidence_valid"
] = (
    df[
        "mean_source_score"
    ]
    >= source_score_median
)


# ------------------------------------------------------------
# 3. Fire evidence
# ------------------------------------------------------------

fire_median = df[
    "total_fire_count"
].median()


df[
    "fire_evidence_valid"
] = (
    df[
        "total_fire_count"
    ]
    >= fire_median
)


# ------------------------------------------------------------
# 4. FRP evidence
# ------------------------------------------------------------

frp_median = df[
    "total_FRP"
].median()


df[
    "frp_evidence_valid"
] = (
    df[
        "total_FRP"
    ]
    >= frp_median
)


# ------------------------------------------------------------
# 5. Wind transport
# ------------------------------------------------------------

wind_median = df[
    "wind_transport_score"
].median()


df[
    "wind_valid"
] = (
    df[
        "wind_transport_score"
    ]
    >= wind_median
)


# ------------------------------------------------------------
# 6. Directional consistency
# ------------------------------------------------------------

direction_median = df[
    "directional_consistency"
].median()


df[
    "direction_valid"
] = (
    df[
        "directional_consistency"
    ]
    >= direction_median
)


# ------------------------------------------------------------
# 7. Downwind enhancement
# ------------------------------------------------------------

df[
    "positive_enhancement"
] = (
    df[
        "observed_mean_enhancement"
    ]
    > 0
)


# ------------------------------------------------------------
# 8. Downwind/upwind ratio
# ------------------------------------------------------------

df[
    "transport_ratio_valid"
] = (
    df[
        "observed_downwind_upwind_ratio"
    ]
    > 1.0
)


# ------------------------------------------------------------
# 9. Statistical significance
# ------------------------------------------------------------

df[
    "statistically_supported"
] = (
    df[
        "fdr_adjusted_p"
    ]
    < 0.05
)


# ------------------------------------------------------------
# 10. Strong statistical significance
# ------------------------------------------------------------

df[
    "strong_statistical_support"
] = (
    df[
        "fdr_adjusted_p"
    ]
    < 0.01
)


# ============================================================
# PHYSICAL CONSISTENCY SCORE
# ============================================================
#
# This is NOT a probability.
#
# It counts independent pieces of supporting evidence.
#
# Maximum = 10
#
# ============================================================

validation_columns = [
    "hdbscan_valid",
    "source_evidence_valid",
    "fire_evidence_valid",
    "frp_evidence_valid",
    "wind_valid",
    "direction_valid",
    "positive_enhancement",
    "transport_ratio_valid",
    "statistically_supported",
    "strong_statistical_support"
]


df[
    "validation_evidence_count"
] = (
    df[
        validation_columns
    ]
    .astype(int)
    .sum(axis=1)
)


# ============================================================
# VALIDATION CLASS
# ============================================================

df[
    "validation_class"
] = "Weak"


# Strong:
# statistically significant,
# positive transport enhancement,
# ratio > 1,
# and at least 7 evidence components.

strong_mask = (
    (df["statistically_supported"])
    &
    (df["positive_enhancement"])
    &
    (df["transport_ratio_valid"])
    &
    (df["validation_evidence_count"] >= 7)
)


df.loc[
    strong_mask,
    "validation_class"
] = "Strong"


# Moderate:
# statistically significant and positive transport,
# but not enough total supporting evidence.

moderate_mask = (
    (~strong_mask)
    &
    (df["statistically_supported"])
    &
    (df["positive_enhancement"])
    &
    (df["transport_ratio_valid"])
)


df.loc[
    moderate_mask,
    "validation_class"
] = "Moderate"


# ============================================================
# FINAL VALIDATION RANK
# ============================================================

df = df.sort_values(
    [
        "validation_class",
        "validation_evidence_count",
        "fdr_adjusted_p",
        "final_source_score"
    ],
    ascending=[
        True,
        False,
        True,
        False
    ]
).copy()


# Explicit class ordering.

class_order = {
    "Strong": 0,
    "Moderate": 1,
    "Weak": 2
}


df[
    "_class_order"
] = df[
    "validation_class"
].map(
    class_order
)


df = df.sort_values(
    [
        "_class_order",
        "validation_evidence_count",
        "fdr_adjusted_p",
        "final_source_score"
    ],
    ascending=[
        True,
        False,
        True,
        False
    ]
).reset_index(
    drop=True
)


df[
    "validation_rank"
] = (
    df.index + 1
)


df = df.drop(
    columns="_class_order"
)


# ============================================================
# SAVE VALIDATION DATASET
# ============================================================

os.makedirs(
    "features",
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY FINAL VALIDATION
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL SOURCE-REGION VALIDATION"
)

print(
    "=" * 70
)


display_columns = [
    "validation_rank",
    "cluster_id",
    "mean_latitude",
    "mean_longitude",
    "cells",
    "mean_source_score",
    "total_fire_count",
    "total_FRP",
    "cluster_persistence",
    "wind_transport_score",
    "directional_consistency",
    "observed_mean_enhancement",
    "observed_downwind_upwind_ratio",
    "effect_size",
    "fdr_adjusted_p",
    "validation_evidence_count",
    "validation_class"
]


print(
    df[
        display_columns
    ].head(20).to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ============================================================
# VALIDATION COUNTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "VALIDATION SUMMARY"
)

print(
    "=" * 70
)


print(
    "Total regions:",
    len(df)
)


print(
    "Strong:",
    (
        df["validation_class"]
        == "Strong"
    ).sum()
)


print(
    "Moderate:",
    (
        df["validation_class"]
        == "Moderate"
    ).sum()
)


print(
    "Weak:",
    (
        df["validation_class"]
        == "Weak"
    ).sum()
)


print(
    "FDR q < 0.05:",
    (
        df["fdr_adjusted_p"]
        < 0.05
    ).sum()
)


print(
    "FDR q < 0.01:",
    (
        df["fdr_adjusted_p"]
        < 0.01
    ).sum()
)


print(
    "Positive downwind enhancement:",
    df[
        "positive_enhancement"
    ].sum()
)


print(
    "Downwind/upwind ratio > 1:",
    df[
        "transport_ratio_valid"
    ].sum()
)


# ============================================================
# STRONG REGIONS
# ============================================================

strong = df[
    df[
        "validation_class"
    ]
    == "Strong"
].copy()


print(
    "\n" + "=" * 70
)

print(
    "STRONG SOURCE REGIONS"
)

print(
    "=" * 70
)


if len(strong) > 0:

    print(
        strong[
            [
                "validation_rank",
                "cluster_id",
                "mean_latitude",
                "mean_longitude",
                "observed_downwind_upwind_ratio",
                "effect_size",
                "fdr_adjusted_p",
                "validation_evidence_count"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

else:

    print(
        "No region met the strong validation criteria."
    )


# Add geographical location metadata
df["location_name"] = df["cluster_id"].apply(lambda cid: get_location_name(cid, "standard"))
df["location_full"] = df["cluster_id"].apply(lambda cid: get_location_name(cid, "full"))
df["location_short"] = df["cluster_id"].apply(lambda cid: get_location_name(cid, "short"))
df["state"] = df["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(int(cid), {}).get("state", "N/A"))
df["country"] = df["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(int(cid), {}).get("country", "India"))
df["zone"] = df["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(int(cid), {}).get("zone", "N/A"))
df["district"] = df["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(int(cid), {}).get("district", "N/A"))
df["region_type"] = df["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(int(cid), {}).get("region_type", "N/A"))

report_columns = [
    "validation_rank",
    "cluster_id",
    "location_name",
    "state",
    "country",
    "zone",
    "district",
    "region_type",
    "mean_latitude",
    "mean_longitude",
    "cells",
    "mean_source_score",
    "total_fire_count",
    "total_FRP",
    "cluster_persistence",
    "mean_wind_speed",
    "resultant_wind_speed",
    "directional_consistency",
    "wind_transport_score",
    "observed_mean_enhancement",
    "observed_downwind_upwind_ratio",
    "effect_size",
    "fdr_adjusted_p",
    "statistical_class",
    "validation_evidence_count",
    "validation_class",
    "final_source_score"
]


report = df[
    [
        c
        for c in report_columns
        if c in df.columns
    ]
].copy()


report.to_csv(
    OUTPUT_REPORT,
    index=False
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL MAP
# ============================================================

print(
    "\nCreating final validation map with location labels..."
)


fig, ax = plt.subplots(
    figsize=(15, 11)
)


# ------------------------------------------------------------
# Plot all regions
# ------------------------------------------------------------

scatter = ax.scatter(
    df["mean_longitude"],
    df["mean_latitude"],
    s=(
        60
        +
        260
        *
        (
            df[
                "validation_evidence_count"
            ]
            /
            10.0
        )
    ),
    c=df["final_source_score"],
    cmap="viridis",
    alpha=0.88,
    edgecolors="black",
    linewidths=0.7,
    zorder=3
)


# ============================================================
# LABEL TOP 16
# ============================================================

for _, row in df.head(
    min(16, len(df))
).iterrows():

    rank_num = int(row['validation_rank'])
    loc_lbl = f"#{rank_num} {row['location_short']}"

    ax.annotate(
        loc_lbl,
        (
            row["mean_longitude"],
            row["mean_latitude"]
        ),
        xytext=(6, 6),
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
    "Longitude (°E)"
)

ax.set_ylabel(
    "Latitude (°N)"
)

ax.set_title(
    "Final Validated HDBSCAN HCHO Source Regions — 2025",
    fontsize=16,
    fontweight="bold"
)


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
    OUTPUT_FILE
)

print(
    OUTPUT_MAP
)

print(
    OUTPUT_REPORT
)


print(
    "\n" + "=" * 70
)

print(
    "FINAL HDBSCAN SOURCE VALIDATION COMPLETE"
)

print(
    "=" * 70
)