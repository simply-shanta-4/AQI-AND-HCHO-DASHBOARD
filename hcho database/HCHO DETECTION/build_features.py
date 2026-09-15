import os
import glob
import warnings

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import rowcol, xy

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

HCHO_DIR = "HCHO"
FIRE_DIR = "FIRE_FRP_2025"
WIND_FILE = "ERA5_wind_2025.nc"

OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "HCHO_FIRE_WIND_2025.csv"
)


# ============================================================
# WIND CALCULATION
# ============================================================

def calculate_wind(u10, v10):
    """
    Calculate wind speed and meteorological wind direction.

    Meteorological direction:
        0   = North
        90  = East
        180 = South
        270 = West

    Direction means the direction FROM which
    the wind is blowing.
    """

    wind_speed = np.sqrt(
        u10 ** 2 +
        v10 ** 2
    )

    wind_direction = (
        np.degrees(
            np.arctan2(
                -u10,
                -v10
            )
        ) + 360
    ) % 360

    return wind_speed, wind_direction


# ============================================================
# LOAD ERA5
# ============================================================

print("\n" + "=" * 60)
print("LOADING ERA5 WIND")
print("=" * 60)

ds = xr.open_dataset(WIND_FILE)

print("Variables:", list(ds.data_vars))
print("Time steps:", len(ds.valid_time))


# ============================================================
# LOAD FIRE DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING FIRE / FRP")
print("=" * 60)

fire_files = sorted(
    glob.glob(
        os.path.join(
            FIRE_DIR,
            "*.csv"
        )
    )
)

if not fire_files:
    raise FileNotFoundError(
        "No CSV file found inside FIRE_FRP_2025/"
    )

fire = pd.read_csv(
    fire_files[0]
)

print(
    "Input fire detections:",
    len(fire)
)


# ============================================================
# CHECK REQUIRED FIRE COLUMNS
# ============================================================

required_columns = [
    "latitude",
    "longitude",
    "acq_date",
    "frp"
]

missing_columns = [
    column
    for column in required_columns
    if column not in fire.columns
]

if missing_columns:

    raise ValueError(
        "Missing fire columns: "
        + str(missing_columns)
    )


# ============================================================
# CLEAN FIRE DATA
# ============================================================

fire["latitude"] = pd.to_numeric(
    fire["latitude"],
    errors="coerce"
)

fire["longitude"] = pd.to_numeric(
    fire["longitude"],
    errors="coerce"
)

fire["frp"] = pd.to_numeric(
    fire["frp"],
    errors="coerce"
)

fire["acq_date"] = pd.to_datetime(
    fire["acq_date"],
    errors="coerce"
)

fire["month"] = (
    fire["acq_date"]
    .dt.month
)

fire = fire.dropna(
    subset=[
        "latitude",
        "longitude",
        "month"
    ]
)

fire["frp"] = fire["frp"].fillna(0)

fire["month"] = (
    fire["month"]
    .astype(int)
)


print(
    "Clean fire detections:",
    len(fire)
)


# ============================================================
# LOAD HCHO FILES
# ============================================================

print("\n" + "=" * 60)
print("PROCESSING HCHO")
print("=" * 60)

hcho_files = sorted(
    glob.glob(
        os.path.join(
            HCHO_DIR,
            "*.tif"
        )
    )
)

if len(hcho_files) != 12:

    raise ValueError(
        f"Expected 12 HCHO files, found {len(hcho_files)}"
    )


all_months = []


# ============================================================
# PROCESS EACH MONTH
# ============================================================

for hcho_file in hcho_files:

    filename = os.path.basename(
        hcho_file
    )

    # --------------------------------------------------------
    # Extract month
    # --------------------------------------------------------

    try:

        month = int(
            filename
            .replace(
                "HCHO_2025_",
                ""
            )
            .replace(
                ".tif",
                ""
            )
        )

    except Exception:

        raise ValueError(
            f"Could not determine month from {filename}"
        )


    print(
        f"\nProcessing month {month:02d}"
    )


    # ========================================================
    # READ HCHO RASTER
    # ========================================================

    with rasterio.open(
        hcho_file
    ) as src:

        hcho = src.read(1)

        transform = src.transform

        height = src.height

        width = src.width

        nodata = src.nodata

        crs = src.crs


    # ========================================================
    # CREATE PIXEL ROW/COLUMN GRID
    # ========================================================

    rows, cols = np.indices(
        (height, width)
    )


    # ========================================================
    # PIXEL CENTER COORDINATES
    # ========================================================

    xs, ys = xy(
        transform,
        rows.ravel(),
        cols.ravel(),
        offset="center"
    )

    longitude = np.asarray(
        xs
    )

    latitude = np.asarray(
        ys
    )

    hcho_values = (
        hcho.ravel()
    )

    row_values = (
        rows.ravel()
    )

    col_values = (
        cols.ravel()
    )


    # ========================================================
    # VALID HCHO PIXELS
    # ========================================================

    valid = np.isfinite(
        hcho_values
    )

    if nodata is not None:

        valid &= (
            hcho_values != nodata
        )

    valid &= (
        hcho_values > -9000
    )


    month_df = pd.DataFrame({

        "month": month,

        "grid_row":
            row_values[valid],

        "grid_col":
            col_values[valid],

        "latitude":
            latitude[valid],

        "longitude":
            longitude[valid],

        "HCHO":
            hcho_values[valid]

    })


    # ========================================================
    # FIRE → ACTUAL HCHO PIXEL ASSIGNMENT
    # ========================================================

    month_fire = fire[
        fire["month"] == month
    ].copy()


    print(
        "Fire detections this month:",
        len(month_fire)
    )


    if len(month_fire) > 0:

        # ----------------------------------------------------
        # Convert every fire coordinate to the actual
        # HCHO raster row/column.
        # ----------------------------------------------------

        fire_rows, fire_cols = rowcol(
            transform,
            month_fire["longitude"].to_numpy(),
            month_fire["latitude"].to_numpy()
        )

        month_fire["grid_row"] = np.asarray(
            fire_rows,
            dtype=int
        )

        month_fire["grid_col"] = np.asarray(
            fire_cols,
            dtype=int
        )


        # ----------------------------------------------------
        # Keep only fires that fall inside the raster
        # ----------------------------------------------------

        inside = (
            (month_fire["grid_row"] >= 0) &
            (month_fire["grid_row"] < height) &
            (month_fire["grid_col"] >= 0) &
            (month_fire["grid_col"] < width)
        )

        month_fire_inside = (
            month_fire.loc[inside]
            .copy()
        )


    else:

        month_fire_inside = month_fire.copy()


    print(
        "Fire detections inside HCHO grid:",
        len(month_fire_inside)
    )


    # ========================================================
    # AGGREGATE FIRE COUNT + FRP PER HCHO PIXEL
    # ========================================================

    if len(month_fire_inside) > 0:

        fire_agg = (
            month_fire_inside
            .groupby(
                [
                    "grid_row",
                    "grid_col"
                ],
                as_index=False
            )
            .agg(

                fire_count=(
                    "latitude",
                    "size"
                ),

                FRP=(
                    "frp",
                    "sum"
                )

            )
        )

    else:

        fire_agg = pd.DataFrame(
            columns=[
                "grid_row",
                "grid_col",
                "fire_count",
                "FRP"
            ]
        )


    # ========================================================
    # MERGE FIRE DATA WITH HCHO PIXELS
    # ========================================================

    month_df = month_df.merge(
        fire_agg,
        on=[
            "grid_row",
            "grid_col"
        ],
        how="left"
    )


    month_df["fire_count"] = (
        month_df["fire_count"]
        .fillna(0)
        .astype(int)
    )

    month_df["FRP"] = (
        month_df["FRP"]
        .fillna(0)
        .astype(float)
    )


    # ========================================================
    # ERA5 WIND
    # ========================================================

    target_time = np.datetime64(
        f"2025-{month:02d}-01"
    )

    time_index = np.argmin(
        np.abs(
            ds.valid_time.values -
            target_time
        )
    )


    u = ds["u10"].isel(
        valid_time=time_index
    )

    v = ds["v10"].isel(
        valid_time=time_index
    )


    # ========================================================
    # INTERPOLATE ERA5 TO HCHO PIXELS
    # ========================================================

    wind_points = xr.Dataset({

        "u10": u,

        "v10": v

    }).interp(

        latitude=xr.DataArray(
            month_df[
                "latitude"
            ].to_numpy(),
            dims="points"
        ),

        longitude=xr.DataArray(
            month_df[
                "longitude"
            ].to_numpy(),
            dims="points"
        ),

        method="nearest"

    )


    u_values = (
        wind_points["u10"]
        .values
    )

    v_values = (
        wind_points["v10"]
        .values
    )


    # ========================================================
    # WIND SPEED + DIRECTION
    # ========================================================

    wind_speed, wind_direction = (
        calculate_wind(
            u_values,
            v_values
        )
    )


    month_df["u10"] = (
        u_values
    )

    month_df["v10"] = (
        v_values
    )

    month_df["wind_speed"] = (
        wind_speed
    )

    month_df["wind_direction"] = (
        wind_direction
    )


    # ========================================================
    # MONTHLY HCHO HOTSPOT THRESHOLD
    # ========================================================

    threshold = (
        month_df["HCHO"]
        .quantile(0.90)
    )


    month_df["HCHO_hotspot"] = (
        month_df["HCHO"] >= threshold
    )


    # ========================================================
    # FIRE SUPPORT
    # ========================================================

    month_df["fire_supported"] = (
        month_df["fire_count"] > 0
    )


    # ========================================================
    # REMOVE GRID INDEX AFTER JOIN
    # ========================================================

    month_df = month_df.drop(
        columns=[
            "grid_row",
            "grid_col"
        ]
    )


    # ========================================================
    # MONTH DIAGNOSTICS
    # ========================================================

    print(
        "Valid HCHO pixels:",
        len(month_df)
    )

    print(
        "HCHO 90th percentile:",
        threshold
    )

    print(
        "HCHO hotspots:",
        int(
            month_df[
                "HCHO_hotspot"
            ].sum()
        )
    )

    print(
        "Fire-supported HCHO hotspots:",
        int(
            (
                month_df[
                    "HCHO_hotspot"
                ]
                &
                month_df[
                    "fire_supported"
                ]
            ).sum()
        )
    )

    print(
        "Aggregated fire detections:",
        int(
            month_df[
                "fire_count"
            ].sum()
        )
    )

    print(
        "Aggregated FRP:",
        float(
            month_df[
                "FRP"
            ].sum()
        )
    )


    all_months.append(
        month_df
    )


# ============================================================
# COMBINE ALL MONTHS
# ============================================================

print("\n" + "=" * 60)
print("COMBINING MONTHS")
print("=" * 60)

features = pd.concat(
    all_months,
    ignore_index=True
)


# ============================================================
# CLEAN
# ============================================================

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

features = features.dropna(
    subset=[
        "HCHO",
        "u10",
        "v10",
        "wind_speed",
        "wind_direction"
    ]
)


# ============================================================
# SORT
# ============================================================

features = features.sort_values(
    [
        "month",
        "latitude",
        "longitude"
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FEATURE DATASET CREATED")
print("=" * 60)

print(
    "Rows:",
    len(features)
)

print(
    "\nColumns:"
)

print(
    features.columns.tolist()
)


print(
    "\nHCHO hotspots:",
    int(
        features[
            "HCHO_hotspot"
        ].sum()
    )
)


print(
    "Fire-supported HCHO hotspots:",
    int(
        (
            features[
                "HCHO_hotspot"
            ]
            &
            features[
                "fire_supported"
            ]
        ).sum()
    )
)


print(
    "\nTotal aggregated fire detections:",
    int(
        features[
            "fire_count"
        ].sum()
    )
)


print(
    "Total aggregated FRP:",
    float(
        features[
            "FRP"
        ].sum()
    )
)


print(
    "\nWind speed range:"
)

print(
    float(
        features[
            "wind_speed"
        ].min()
    ),
    "to",
    float(
        features[
            "wind_speed"
        ].max()
    )
)


print(
    "\nWind direction range:"
)

print(
    float(
        features[
            "wind_direction"
        ].min()
    ),
    "to",
    float(
        features[
            "wind_direction"
        ].max()
    )
)


print(
    "\nMissing values:"
)

print(
    features[
        [
            "HCHO",
            "fire_count",
            "FRP",
            "u10",
            "v10",
            "wind_speed",
            "wind_direction"
        ]
    ]
    .isna()
    .sum()
)


print(
    "\nSaved:",
    OUTPUT_FILE
)


print(
    "\n" + "=" * 60
)

print(
    "BUILD COMPLETE"
)

print(
    "=" * 60
)