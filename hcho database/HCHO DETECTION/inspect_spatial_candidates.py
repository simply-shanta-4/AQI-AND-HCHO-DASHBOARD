import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "features/SPATIAL_SOURCE_CANDIDATES_2025.csv"
)

OUTPUT_DIR = "features"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "SPATIAL_CANDIDATE_INSPECTION_2025.csv"
)

OUTPUT_MAP = os.path.join(
    OUTPUT_DIR,
    "SPATIAL_CANDIDATE_INSPECTION_2025.png"
)


# ============================================================
# LOAD
# ============================================================

print("\n" + "=" * 60)
print("SPATIAL CANDIDATE INSPECTION")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(
    "Candidates:",
    len(df)
)


# ============================================================
# BASIC STATISTICS
# ============================================================

print(
    "\n========== GEOGRAPHIC RANGE =========="
)

print(
    "Latitude:",
    f"{df['latitude'].min():.3f}",
    "to",
    f"{df['latitude'].max():.3f}"
)

print(
    "Longitude:",
    f"{df['longitude'].min():.3f}",
    "to",
    f"{df['longitude'].max():.3f}"
)


# ============================================================
# ACTIVE MONTH STATISTICS
# ============================================================

print(
    "\n========== ACTIVE MONTHS =========="
)

print(
    "Mean:",
    f"{df['active_months'].mean():.3f}"
)

print(
    "Median:",
    f"{df['active_months'].median():.1f}"
)

print(
    "Maximum:",
    int(
        df['active_months'].max()
    )
)

print(
    "\nDistribution:"
)

print(
    df[
        "active_months"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# SOURCE SCORE
# ============================================================

print(
    "\n========== SOURCE SCORE =========="
)

print(
    "Mean:",
    f"{df['mean_source_score'].mean():.4f}"
)

print(
    "Median:",
    f"{df['mean_source_score'].median():.4f}"
)

print(
    "90th percentile:",
    f"{df['mean_source_score'].quantile(0.90):.4f}"
)

print(
    "95th percentile:",
    f"{df['mean_source_score'].quantile(0.95):.4f}"
)

print(
    "99th percentile:",
    f"{df['mean_source_score'].quantile(0.99):.4f}"
)


# ============================================================
# CREATE SCORE GROUPS
# ============================================================

df["score_percentile"] = (
    df[
        "mean_source_score"
    ]
    .rank(
        pct=True
    )
)


df["score_group"] = pd.cut(

    df["score_percentile"],

    bins=[
        0,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00
    ],

    labels=[
        "0-50%",
        "50-75%",
        "75-90%",
        "90-95%",
        "95-99%",
        "99-100%"
    ],

    include_lowest=True
)


print(
    "\n========== SCORE GROUPS =========="
)

print(
    df[
        "score_group"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# PERSISTENT CANDIDATES
# ============================================================

print(
    "\n========== PERSISTENCE CHECK =========="
)

for threshold in [
    2,
    3,
    4,
    5,
    6
]:

    count = int(
        (
            df[
                "active_months"
            ] >= threshold
        ).sum()
    )

    percentage = (
        100
        * count
        / len(df)
    )

    print(
        f"Active >= {threshold} months:"
        f" {count} ({percentage:.2f}%)"
    )


# ============================================================
# HIGH-SCORE CANDIDATES
# ============================================================

print(
    "\n========== HIGH-SCORE CANDIDATES =========="
)

for percentile in [
    0.90,
    0.95,
    0.99
]:

    threshold = (
        df[
            "mean_source_score"
        ]
        .quantile(
            percentile
        )
    )

    subset = df[
        df[
            "mean_source_score"
        ] >= threshold
    ]

    print(
        f"\nTop {int((1-percentile)*100)}%:"
    )

    print(
        "Count:",
        len(subset)
    )

    print(
        "Mean active months:",
        f"{subset['active_months'].mean():.3f}"
    )

    print(
        "Mean HCHO:",
        f"{subset['mean_HCHO'].mean():.6f}"
    )

    print(
        "Mean fire count:",
        f"{subset['total_fire_count'].mean():.2f}"
    )

    print(
        "Mean FRP:",
        f"{subset['total_FRP'].mean():.2f}"
    )


# ============================================================
# SPATIAL MAP — ALL CANDIDATES
# ============================================================

print(
    "\n========== CREATING MAP =========="
)

plt.figure(
    figsize=(12, 8)
)

scatter = plt.scatter(

    df["longitude"],

    df["latitude"],

    c=df["mean_source_score"],

    s=8,

    alpha=0.55
)

plt.colorbar(
    scatter,
    label="Mean source score"
)

plt.xlabel(
    "Longitude (°E)"
)

plt.ylabel(
    "Latitude (°N)"
)

plt.title(
    "Spatial Distribution of HCHO Source Candidates — 2025"
)

plt.xlim(
    68,
    97.5
)

plt.ylim(
    6,
    37.5
)

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout()


plt.savefig(
    OUTPUT_MAP,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE INSPECTION DATA
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print(
    "\nSaved map:"
)

print(
    OUTPUT_MAP
)

print(
    "\nSaved inspection dataset:"
)

print(
    OUTPUT_CSV
)

print(
    "\n" + "=" * 60
)

print(
    "SPATIAL INSPECTION COMPLETE"
)

print(
    "=" * 60
)