import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Handle path for location_metadata import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from location_metadata import CLUSTER_LOCATIONS, get_location_name, INDIAN_CLUSTER_IDS
except ImportError:
    # Fallback if imported from parent directory
    from location_metadata import CLUSTER_LOCATIONS, get_location_name, INDIAN_CLUSTER_IDS


# ============================================================
# FINAL FIGURES AND TABLES
# ============================================================

features_candidates = [
    os.path.join(current_dir, "features"),
    os.path.join(current_dir, "HCHO DETECTION", "features"),
    "features"
]
FEATURES_DIR = "features"
for fc in features_candidates:
    if os.path.exists(fc):
        FEATURES_DIR = fc
        break

VALIDATION_FILE = os.path.join(FEATURES_DIR, "FINAL_SOURCE_REGION_VALIDATION_2025.csv")
REPORT_FILE = os.path.join(FEATURES_DIR, "FINAL_SOURCE_REGION_REPORT_2025.csv")
TRANSPORT_FILE = os.path.join(FEATURES_DIR, "SOURCE_RECEPTOR_TRANSPORT_MONTHLY_2025.csv")

# ============================================================
# OUTPUTS
# ============================================================

FINAL_TABLE = os.path.join(FEATURES_DIR, "FINAL_HDBSCAN_SOURCE_TABLE_2025.csv")
SUMMARY_TABLE = os.path.join(FEATURES_DIR, "FINAL_SOURCE_REGION_SUMMARY_2025.csv")
MAP_FILE = os.path.join(FEATURES_DIR, "FINAL_SOURCE_REGION_MAP_2025.png")
RANKING_FILE = os.path.join(FEATURES_DIR, "FINAL_SOURCE_REGION_RANKING_2025.png")
TRANSPORT_FIGURE = os.path.join(FEATURES_DIR, "FINAL_TRANSPORT_EVIDENCE_2025.png")
ENHANCEMENT_FILE = os.path.join(FEATURES_DIR, "FINAL_DOWNWIND_ENHANCEMENT_2025.png")


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("FINAL HDBSCAN FIGURES AND TABLES (DOMESTIC INDIA ONLY)")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [VALIDATION_FILE, REPORT_FILE, TRANSPORT_FILE]
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"\nRequired file not found:\n{file}")


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading final validation data...")
validation = pd.read_csv(VALIDATION_FILE)
validation["cluster_id"] = validation["cluster_id"].astype(int)
validation = validation[validation["cluster_id"].isin(INDIAN_CLUSTER_IDS)].copy().sort_values("validation_rank").reset_index(drop=True)
validation["validation_rank"] = range(1, len(validation) + 1)
print("Domestic Indian Validation regions:", len(validation))

# ============================================================
# LOAD REPORT
# ============================================================

print("\nLoading final report data...")
report = pd.read_csv(REPORT_FILE)
report["cluster_id"] = report["cluster_id"].astype(int)
report = report[report["cluster_id"].isin(INDIAN_CLUSTER_IDS)].copy().sort_values("validation_rank").reset_index(drop=True)
report["validation_rank"] = range(1, len(report) + 1)
print("Domestic Indian Report regions:", len(report))

# ============================================================
# LOAD TRANSPORT DATA
# ============================================================

print("\nLoading transport data...")
transport = pd.read_csv(TRANSPORT_FILE)
transport["cluster_id"] = transport["cluster_id"].astype(int)
transport = transport[transport["cluster_id"].isin(INDIAN_CLUSTER_IDS)].copy().reset_index(drop=True)
print("Domestic Indian Transport records:", len(transport))

# ============================================================
# CLEAN & ENRICH WITH GEOGRAPHICAL METADATA
# ============================================================

for data in [validation, report, transport]:
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    if "cluster_id" in data.columns:
        data["location_name"] = data["cluster_id"].apply(lambda cid: get_location_name(cid, "standard"))
        data["location_full"] = data["cluster_id"].apply(lambda cid: get_location_name(cid, "full"))
        data["location_short"] = data["cluster_id"].apply(lambda cid: get_location_name(cid, "short"))
        data["state"] = data["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(cid, {}).get("state", "N/A"))
        data["country"] = data["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(cid, {}).get("country", "India"))
        data["zone"] = data["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(cid, {}).get("zone", "N/A"))
        data["district"] = data["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(cid, {}).get("district", "N/A"))
        data["region_type"] = data["cluster_id"].apply(lambda cid: CLUSTER_LOCATIONS.get(cid, {}).get("region_type", "N/A"))



# ============================================================
# FINAL TABLE
# ============================================================

print("\nCreating final source-region table...")

table_columns = [
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

table_columns = [c for c in table_columns if c in validation.columns]
final_table = validation[table_columns].copy().sort_values("validation_rank")
final_table.to_csv(FINAL_TABLE, index=False)
print("Saved:", FINAL_TABLE)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_columns = [
    "validation_rank",
    "cluster_id",
    "location_name",
    "state",
    "zone",
    "region_type",
    "mean_latitude",
    "mean_longitude",
    "observed_downwind_upwind_ratio",
    "effect_size",
    "fdr_adjusted_p",
    "validation_evidence_count",
    "validation_class",
    "final_source_score"
]

summary_columns = [c for c in summary_columns if c in validation.columns]
summary = validation[summary_columns].copy().sort_values("validation_rank")
summary.to_csv(SUMMARY_TABLE, index=False)
print("Saved:", SUMMARY_TABLE)


# ============================================================
# FIGURE 1: FINAL SOURCE REGION MAP
# ============================================================

print("\nCreating final source-region map with location labels...")

fig, ax = plt.subplots(figsize=(15, 11))

scatter = ax.scatter(
    validation["mean_longitude"],
    validation["mean_latitude"],
    s=(60 + 260 * validation["validation_evidence_count"] / 10.0),
    c=validation["final_source_score"],
    cmap="viridis",
    edgecolors="black",
    linewidths=0.7,
    alpha=0.88,
    zorder=3
)

# Label top 16 regions with location names
top_regions = validation.head(min(16, len(validation)))

for _, row in top_regions.iterrows():
    rank_num = int(row['validation_rank'])
    loc_lbl = f"#{rank_num} {row['location_short']}"
    
    ax.annotate(
        loc_lbl,
        (row["mean_longitude"], row["mean_latitude"]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.82, edgecolor="#CBD5E1", linewidth=0.5),
        zorder=4
    )

ax.set_xlabel("Longitude (°E)", fontsize=12, fontweight="bold")
ax.set_ylabel("Latitude (°N)", fontsize=12, fontweight="bold")
ax.set_title("Validated Domestic Indian HCHO Hotspot Source Regions — 2025", fontsize=16, fontweight="bold", pad=14)
ax.set_xlim(68, 96)
ax.set_ylim(7.5, 36.5)
ax.grid(True, alpha=0.3, linestyle="--", zorder=1)

cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Final Source Evidence Score", fontsize=11, fontweight="bold")


plt.tight_layout()
plt.savefig(MAP_FILE, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", MAP_FILE)


# ============================================================
# FIGURE 2: SOURCE REGION RANKING (WITH LOCATION NAMES)
# ============================================================

print("\nCreating source-region ranking figure with location names...")

ranking = validation.head(min(15, len(validation))).copy()
ranking = ranking.sort_values("validation_rank", ascending=True)

fig, ax = plt.subplots(figsize=(13, 8.5))

y = np.arange(len(ranking))

# Color bars according to validation confidence
colors = [
    "#10B981" if vc == "Strong" else ("#F59E0B" if vc == "Moderate" else "#EF4444")
    for vc in ranking["validation_class"]
]

bars = ax.barh(y, ranking["final_source_score"], color=colors, edgecolor="#1E293B", linewidth=0.5, height=0.68)

# Add score value text at the end of each bar
for i, bar in enumerate(bars):
    width_val = bar.get_width()
    ax.text(width_val + 0.012, bar.get_y() + bar.get_height()/2, f"{width_val:.3f}", 
            va='center', ha='left', fontsize=9, fontweight="bold", color="#1F2937")

ax.set_yticks(y)
# Use clean location names with rank for Y-tick labels
y_labels = [f"#{int(r['validation_rank'])} {r['location_full']}" for _, r in ranking.iterrows()]
ax.set_yticklabels(y_labels, fontsize=10.5, fontweight="500")

ax.invert_yaxis()
ax.set_xlabel("Final Source Evidence Score", fontsize=12, fontweight="bold")
ax.set_ylabel("Source Region Location", fontsize=12, fontweight="bold")
ax.set_title("Top Validated HCHO Hotspot Regions by Location — 2025", fontsize=15, fontweight="bold", pad=12)
ax.set_xlim(0, max(ranking["final_source_score"]) * 1.15)
ax.grid(axis="x", alpha=0.3, linestyle="--")

# Custom Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#10B981', edgecolor='#1E293B', label='Strong Validation'),
    Patch(facecolor='#F59E0B', edgecolor='#1E293B', label='Moderate Validation'),
    Patch(facecolor='#EF4444', edgecolor='#1E293B', label='Weak / Diffuse')
]
ax.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1")

plt.tight_layout()
plt.savefig(RANKING_FILE, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", RANKING_FILE)


# ============================================================
# FIGURE 3: TRANSPORT EVIDENCE (WITH LOCATION NAMES)
# ============================================================

print("\nCreating transport evidence figure with location names...")

transport_summary = (
    transport
    .groupby("cluster_id", as_index=False)
    .agg(
        mean_downwind_HCHO=("mean_downwind_HCHO", "mean"),
        mean_upwind_HCHO=("mean_upwind_HCHO", "mean"),
        mean_ratio=("downwind_upwind_ratio", "mean")
    )
)

transport_summary = transport_summary.merge(
    validation[["cluster_id", "validation_rank", "validation_class", "location_short", "location_name"]],
    on="cluster_id",
    how="left"
)

transport_summary = transport_summary.sort_values("validation_rank").head(min(12, len(transport_summary)))

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(transport_summary))
width = 0.36

rects1 = ax.bar(x - width / 2, transport_summary["mean_downwind_HCHO"], width, label="Downwind HCHO (Plume)", color="#EF4444", alpha=0.9, edgecolor="#7F1D1D", linewidth=0.5)
rects2 = ax.bar(x + width / 2, transport_summary["mean_upwind_HCHO"], width, label="Upwind HCHO (Baseline)", color="#10B981", alpha=0.9, edgecolor="#064E3B", linewidth=0.5)

ax.set_xticks(x)
x_labels = [f"#{int(r['validation_rank'])}\n{r['location_short']}" for _, r in transport_summary.iterrows()]
ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=9.5, fontweight="600")

ax.set_ylabel("Mean HCHO Column Density (mol/m²)", fontsize=12, fontweight="bold")
ax.set_xlabel("Source Region Location", fontsize=12, fontweight="bold")
ax.set_title("Downwind Plume vs Upwind Background HCHO across Top Locations — 2025", fontsize=15, fontweight="bold", pad=12)
ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=11)
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(TRANSPORT_FIGURE, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", TRANSPORT_FIGURE)


# ============================================================
# FIGURE 4: DOWNWIND / UPWIND RATIO (WITH LOCATION NAMES)
# ============================================================

print("\nCreating transport-ratio figure with location names...")

ratio_data = validation.head(min(15, len(validation))).copy()
ratio_data = ratio_data.sort_values("validation_rank", ascending=True)

fig, ax = plt.subplots(figsize=(13, 8.5))

y = np.arange(len(ratio_data))

ratio_colors = [
    "#059669" if r >= 1.20 else ("#10B981" if r >= 1.05 else ("#F59E0B" if r >= 1.0 else "#EF4444"))
    for r in ratio_data["observed_downwind_upwind_ratio"]
]

bars = ax.barh(y, ratio_data["observed_downwind_upwind_ratio"], color=ratio_colors, edgecolor="#1E293B", linewidth=0.5, height=0.68)

# Add ratio text label
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.015, bar.get_y() + bar.get_height()/2, f"{w:.3f}x", 
            va='center', ha='left', fontsize=9, fontweight="bold", color="#1F2937")

ax.axvline(1.0, linestyle="--", color="#DC2626", linewidth=1.6, label="Baseline Ratio = 1.0 (No Enhancement)")

ax.set_yticks(y)
y_ratio_labels = [f"#{int(r['validation_rank'])} {r['location_full']}" for _, r in ratio_data.iterrows()]
ax.set_yticklabels(y_ratio_labels, fontsize=10.5, fontweight="500")

ax.invert_yaxis()
ax.set_xlabel("Observed Downwind / Upwind HCHO Ratio", fontsize=12, fontweight="bold")
ax.set_ylabel("Source Region Location", fontsize=12, fontweight="bold")
ax.set_title("Downwind HCHO Plume Enhancement Ratio by Location — 2025", fontsize=15, fontweight="bold", pad=12)
ax.set_xlim(0.85, max(ratio_data["observed_downwind_upwind_ratio"]) * 1.12)
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1")

plt.tight_layout()
plt.savefig(ENHANCEMENT_FILE, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", ENHANCEMENT_FILE)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL FIGURES AND TABLES COMPLETE (LOCATION NAMES INCLUDED)")
print("=" * 70)
print(f"Files updated:\n{FINAL_TABLE}\n{SUMMARY_TABLE}\n{MAP_FILE}\n{RANKING_FILE}\n{TRANSPORT_FIGURE}\n{ENHANCEMENT_FILE}")