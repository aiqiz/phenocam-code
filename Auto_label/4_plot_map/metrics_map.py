import os
import json
import pickle
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path
import numpy as np


csv_path = '/Users/aiqizhang/Desktop/traffic/Traffic Camera List - 4326.csv'
df = pd.read_csv(csv_path)
df['REC_ID'] = df['REC_ID'].astype(str).str.strip().str.lower()

rec_ids, lats, lons = [], [], []
for _, row in df.iterrows():
    rec_id = row['REC_ID']
    geometry_str = str(row['geometry'])
    try:
        geometry = json.loads(geometry_str.replace("'", '"'))
        coords = geometry.get("coordinates", [])
        if coords and isinstance(coords[0], list) and len(coords[0]) == 2:
            lon, lat = coords[0]
            rec_ids.append(rec_id)
            lats.append(lat)
            lons.append(lon)
    except Exception as e:
        print(f"[Error] geometry parse failed for {rec_id}: {e}")

df_locations = pd.DataFrame({
    "REC_ID": rec_ids,
    "lat": lats,
    "lon": lons
})


ads_results_dir = Path("/Users/aiqizhang/Desktop/traffic/map_prepare_2024")  # UPDATE
records = []

for pkl_file in ads_results_dir.glob("loc*.pkl"):
    loc_id = pkl_file.stem.lower().replace("loc", "")
    with open(pkl_file, "rb") as f:
        results = pickle.load(f)

    metrics = results.get("Tree", {}).get("metrics", {}) # or "Grass"

    sos, eos, los, peak_date, peak_val = (
        metrics.get("SOS"),
        metrics.get("EOS"),
        metrics.get("LOS"),
        metrics.get("PeakDate"),
        metrics.get("PeakValue"),
    )

    # Skip if no SOS or SOS < 50 DOY
    if eos is not None and sos is not None:
        sos_doy = sos.day_of_year
        eos_doy = eos.day_of_year
        if los >= 80 and eos_doy <= 330:
        # and eos_doy <= 305 and eos_doy >= 275
            rec = {"REC_ID": loc_id,
                   "SOS": sos,
                   "SOS_doy": sos.dayofyear}

            if eos is not None:
                eos = pd.to_datetime(eos)
                rec["EOS"] = eos
                rec["EOS_doy"] = eos.dayofyear

            if los is not None:
                rec["LOS"] = los

            if peak_date is not None:
                peak_date = pd.to_datetime(peak_date)
                rec["PeakDate"] = peak_date
                rec["Peak_doy"] = peak_date.dayofyear
                rec["PeakVal"] = peak_val

            records.append(rec)

df_metrics = pd.DataFrame(records)


df_merge = pd.merge(df_locations, df_metrics, on="REC_ID", how="inner")

gdf = gpd.GeoDataFrame(
    df_merge,
    geometry=gpd.points_from_xy(df_merge['lon'], df_merge['lat']),
    crs="EPSG:4326"
).to_crs(epsg=3857)


plot_column = "EOS_doy"   #SOS_doy, LOS, Peak_doy...

fig, ax = plt.subplots(figsize=(10, 10))
# warm–cold colorbar
cmap = "coolwarm"

vmin = gdf[plot_column].min()
vmax = gdf[plot_column].max()

gdf.plot(
    ax=ax,
    column=plot_column,
    cmap=cmap,
    vmin=vmin, vmax=vmax,
    legend=True,
    legend_kwds={"label": f"{plot_column} (DOY)", "shrink": 0.85},
    markersize=150,
    alpha=0.9,
    edgecolor="k"
)


ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_title(f"Traffic Cameras Tree — {plot_column}")
ax.set_axis_off()
plt.tight_layout()
plt.show()

print(f"Mapped {len(gdf)} locations with {plot_column}")
