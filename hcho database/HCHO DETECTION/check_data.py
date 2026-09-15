import os
import glob
import pandas as pd
import rasterio
import xarray as xr


print("\n" + "=" * 50)
print("DATASET VERIFICATION")
print("=" * 50)


# ============================================================
# 1. HCHO
# ============================================================

print("\n========== HCHO ==========")

hcho_files = sorted(
    glob.glob("HCHO/*.tif")
)

print(
    "Number of HCHO files:",
    len(hcho_files)
)

if len(hcho_files) == 0:
    raise FileNotFoundError(
        "No HCHO GeoTIFF files found in HCHO/"
    )

with rasterio.open(hcho_files[0]) as src:

    print(
        "First file:",
        hcho_files[0]
    )

    print(
        "Shape:",
        src.height,
        "x",
        src.width
    )

    print(
        "CRS:",
        src.crs
    )

    print(
        "Bounds:",
        src.bounds
    )

    print(
        "NoData:",
        src.nodata
    )


# ============================================================
# 2. FIRE + FRP
# ============================================================

print("\n========== FIRE + FRP ==========")

fire_files = sorted(
    glob.glob(
        "FIRE_FRP_2025/*.csv"
    )
)

if len(fire_files) == 0:
    raise FileNotFoundError(
        "No CSV file found in FIRE_FRP_2025/"
    )

print(
    "Fire files:",
    len(fire_files)
)

fire_df = pd.read_csv(
    fire_files[0]
)

print(
    "Rows:",
    len(fire_df)
)

print(
    "Columns:"
)

print(
    fire_df.columns.tolist()
)


# ============================================================
# 3. ERA5 WIND
# ============================================================

print("\n========== ERA5 WIND ==========")

wind_file = (
    "ERA5_wind_2025.nc"
)

if not os.path.exists(wind_file):

    raise FileNotFoundError(
        f"Missing {wind_file}"
    )

ds = xr.open_dataset(
    wind_file
)

print(ds)

print(
    "\nVariables:"
)

print(
    list(ds.data_vars)
)

print(
    "\nCoordinates:"
)

print(
    list(ds.coords)
)

ds.close()


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 50
)

print(
    "DATASET VERIFICATION COMPLETE"
)

print(
    "=" * 50
)