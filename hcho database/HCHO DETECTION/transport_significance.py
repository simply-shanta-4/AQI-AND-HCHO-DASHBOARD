import os
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    "features/SOURCE_RECEPTOR_TRANSPORT_MONTHLY_2025.csv"
)

OUTPUT_FILE = (
    "features/SOURCE_RECEPTOR_TRANSPORT_SIGNIFICANCE_2025.csv"
)

N_PERMUTATIONS = 1000

RANDOM_SEED = 42


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 60)
print("SOURCE → RECEPTOR STATISTICAL SIGNIFICANCE")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading transport results...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Monthly transport records:",
    len(df)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "cluster_id",
    "month",
    "mean_downwind_HCHO",
    "mean_upwind_HCHO",
    "downwind_upwind_ratio",
    "transport_direction",
    "downwind_cells",
    "upwind_cells"
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
# CLEAN DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=[
        "cluster_id",
        "month",
        "mean_downwind_HCHO",
        "mean_upwind_HCHO",
        "transport_direction"
    ]
).copy()


print(
    "Valid transport records:",
    len(df)
)


# ============================================================
# OBSERVED ENHANCEMENT
# ============================================================

df[
    "observed_enhancement"
] = (
    df["mean_downwind_HCHO"]
    -
    df["mean_upwind_HCHO"]
)


# ============================================================
# OBSERVED RATIO
# ============================================================

df[
    "observed_ratio"
] = np.where(

    df["mean_upwind_HCHO"] > 0,

    df["mean_downwind_HCHO"]
    /
    df["mean_upwind_HCHO"],

    np.nan
)


# ============================================================
# RANDOM NUMBER GENERATOR
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# REGION ANALYSIS
# ============================================================

results = []

cluster_ids = sorted(
    df["cluster_id"].unique()
)

print(
    "\nHDBSCAN regions:",
    len(cluster_ids)
)

print(
    "Permutations per region:",
    N_PERMUTATIONS
)


for counter, cluster_id in enumerate(
    cluster_ids,
    start=1
):

    region = df[
        df["cluster_id"] == cluster_id
    ].copy()


    region = region[
        np.isfinite(
            region[
                "observed_enhancement"
            ]
        )
    ]


    if len(region) < 2:

        print(
            f"Region {counter}/{len(cluster_ids)} "
            f"| Cluster {cluster_id} "
            f"| insufficient data"
        )

        continue


    # ========================================================
    # OBSERVED VALUES
    # ========================================================

    observed = region[
        "observed_enhancement"
    ].to_numpy(
        dtype=float
    )


    # ========================================================
    # OBSERVED MEAN
    # ========================================================

    observed_mean = float(
        np.mean(
            observed
        )
    )


    # ========================================================
    # OBSERVED MEDIAN
    # ========================================================

    observed_median = float(
        np.median(
            observed
        )
    )


    # ========================================================
    # POSITIVE MONTH FRACTION
    # ========================================================

    positive_fraction = float(
        np.mean(
            observed > 0
        )
    )


    # ========================================================
    # OBSERVED DOWNWIND / UPWIND RATIO
    # ========================================================

    ratios = region[
        "observed_ratio"
    ].to_numpy(
        dtype=float
    )

    ratios = ratios[
        np.isfinite(
            ratios
        )
    ]


    if len(ratios) > 0:

        observed_ratio = float(
            np.mean(
                ratios
            )
        )

    else:

        observed_ratio = np.nan


    # ========================================================
    # PERMUTATION TEST
    # ========================================================
    #
    # Null hypothesis:
    #
    # There is no systematic preference for positive
    # downwind-minus-upwind HCHO enhancement.
    #
    # We randomly reverse the sign of each monthly
    # enhancement.
    #
    # ========================================================

    null_means = np.empty(
        N_PERMUTATIONS,
        dtype=float
    )


    for i in range(
        N_PERMUTATIONS
    ):

        random_signs = rng.choice(
            [-1.0, 1.0],
            size=len(observed)
        )


        randomized = (
            observed
            *
            random_signs
        )


        null_means[i] = np.mean(
            randomized
        )


    # ========================================================
    # TWO-SIDED P-VALUE
    # ========================================================

    two_sided_count = np.sum(
        np.abs(
            null_means
        )
        >=
        abs(
            observed_mean
        )
    )


    p_value_two_sided = (
        two_sided_count + 1
    ) / (
        N_PERMUTATIONS + 1
    )


    # ========================================================
    # ONE-SIDED P-VALUE
    #
    # Alternative:
    #
    # downwind HCHO > upwind HCHO
    #
    # ========================================================

    positive_count = np.sum(
        null_means
        >=
        observed_mean
    )


    p_value_positive = (
        positive_count + 1
    ) / (
        N_PERMUTATIONS + 1
    )


    # ========================================================
    # NULL DISTRIBUTION
    # ========================================================

    null_mean = float(
        np.mean(
            null_means
        )
    )


    null_std = float(
        np.std(
            null_means,
            ddof=1
        )
    )


    null_lower = float(
        np.percentile(
            null_means,
            2.5
        )
    )


    null_upper = float(
        np.percentile(
            null_means,
            97.5
        )
    )


    # ========================================================
    # EFFECT SIZE
    # ========================================================

    if null_std > 0:

        effect_size = (
            observed_mean
            /
            null_std
        )

    else:

        effect_size = np.nan


    # ========================================================
    # UNCORRECTED SIGNIFICANCE
    # ========================================================

    significant_05 = (
        observed_mean > 0
        and
        p_value_positive < 0.05
    )


    significant_01 = (
        observed_mean > 0
        and
        p_value_positive < 0.01
    )


    # ========================================================
    # STORE
    # ========================================================

    results.append({

        "cluster_id":
            int(cluster_id),

        "months":
            len(region),

        "observed_mean_enhancement":
            observed_mean,

        "observed_median_enhancement":
            observed_median,

        "observed_downwind_upwind_ratio":
            observed_ratio,

        "positive_month_fraction":
            positive_fraction,

        "null_mean":
            null_mean,

        "null_std":
            null_std,

        "null_2_5_percentile":
            null_lower,

        "null_97_5_percentile":
            null_upper,

        "effect_size":
            effect_size,

        "p_value_two_sided":
            p_value_two_sided,

        "p_value_downwind_positive":
            p_value_positive,

        "significant_05":
            significant_05,

        "significant_01":
            significant_01
    })


    print(
        f"Region {counter}/{len(cluster_ids)} "
        f"| Cluster {cluster_id} "
        f"| p = {p_value_positive:.4f}"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    results
)


if len(results) == 0:

    raise RuntimeError(
        "No valid regions were available "
        "for statistical testing."
    )


# ============================================================
# BENJAMINI-HOCHBERG FDR CORRECTION
# ============================================================
#
# We test multiple HDBSCAN source regions.
#
# FDR correction controls the expected proportion of
# false discoveries among regions called significant.
#
# ============================================================

print(
    "\nApplying Benjamini-Hochberg FDR correction..."
)


# Sort by raw one-sided p-value.

results = results.sort_values(
    "p_value_downwind_positive",
    ascending=True
).reset_index(
    drop=True
)


m = len(results)


# Original BH values.

results[
    "fdr_rank"
] = (
    np.arange(m) + 1
)


results[
    "fdr_adjusted_p_raw"
] = (

    results[
        "p_value_downwind_positive"
    ]
    *
    m
    /
    results[
        "fdr_rank"
    ]
)


# ------------------------------------------------------------
# IMPORTANT:
# Make an explicit writable NumPy copy.
#
# This fixes the "assignment destination is read-only"
# error encountered previously.
# ------------------------------------------------------------

adjusted = np.array(
    results[
        "fdr_adjusted_p_raw"
    ],
    dtype=float,
    copy=True
)


# ------------------------------------------------------------
# Enforce monotonicity from largest p-value backwards.
# ------------------------------------------------------------

for i in range(
    len(adjusted) - 2,
    -1,
    -1
):

    adjusted[i] = min(
        adjusted[i],
        adjusted[i + 1]
    )


# Limit to [0, 1].

adjusted = np.clip(
    adjusted,
    0.0,
    1.0
)


results[
    "fdr_adjusted_p"
] = adjusted


# ============================================================
# FDR SIGNIFICANCE
# ============================================================

results[
    "fdr_significant_05"
] = (

    (
        results[
            "fdr_adjusted_p"
        ]
        <
        0.05
    )

    &

    (
        results[
            "observed_mean_enhancement"
        ]
        >
        0
    )
)


results[
    "fdr_significant_01"
] = (

    (
        results[
            "fdr_adjusted_p"
        ]
        <
        0.01
    )

    &

    (
        results[
            "observed_mean_enhancement"
        ]
        >
        0
    )
)


# ============================================================
# FINAL STATISTICAL RANK
# ============================================================

results = results.sort_values(

    [
        "fdr_significant_05",
        "fdr_adjusted_p",
        "p_value_downwind_positive",
        "observed_mean_enhancement"
    ],

    ascending=[
        False,
        True,
        True,
        False
    ]

).reset_index(
    drop=True
)


results[
    "statistical_rank"
] = (
    results.index + 1
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "features",
    exist_ok=True
)


results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# COMPLETE RESULTS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "STATISTICAL SIGNIFICANCE RESULTS"
)

print(
    "=" * 60
)


display_columns = [

    "statistical_rank",

    "cluster_id",

    "months",

    "observed_mean_enhancement",

    "observed_downwind_upwind_ratio",

    "positive_month_fraction",

    "effect_size",

    "p_value_downwind_positive",

    "fdr_adjusted_p",

    "fdr_significant_05",

    "fdr_significant_01"
]


print(
    results[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ============================================================
# SIGNIFICANT REGIONS
# ============================================================

significant = results[
    results[
        "fdr_significant_05"
    ]
].copy()


print(
    "\n" + "=" * 60
)

print(
    "STATISTICALLY SUPPORTED REGIONS"
)

print(
    "=" * 60
)


print(
    "FDR-adjusted p < 0.05:",
    len(significant)
)


if len(significant) > 0:

    print(
        significant[
            [
                "statistical_rank",
                "cluster_id",
                "months",
                "observed_mean_enhancement",
                "observed_downwind_upwind_ratio",
                "effect_size",
                "p_value_downwind_positive",
                "fdr_adjusted_p"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
        )
    )

else:

    print(
        "No HDBSCAN source region passed "
        "the FDR-adjusted 0.05 threshold."
    )


# ============================================================
# VERY STRONG REGIONS
# ============================================================

very_strong = results[
    results[
        "fdr_significant_01"
    ]
].copy()


print(
    "\nFDR-adjusted p < 0.01:",
    len(very_strong)
)


if len(very_strong) > 0:

    print(
        very_strong[
            [
                "cluster_id",
                "observed_mean_enhancement",
                "observed_downwind_upwind_ratio",
                "effect_size",
                "fdr_adjusted_p"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}"
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
    "TRANSPORT SIGNIFICANCE ANALYSIS COMPLETE"
)

print(
    "=" * 60
)