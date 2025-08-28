import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import PillowWriter
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap

from scipy.interpolate import Rbf, griddata
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

CSV_PATH = "/Users/aiqizhang/Desktop/traffic/Traffic Camera List - 4326.csv"
PKL_DIR  = "/Users/aiqizhang/Desktop/traffic/map_prepare_2023"
OUTPUT_GIF = "field_2023_test.gif"
PHENO_CLASS = "Tree"   # or "Grass"

df = pd.read_csv(CSV_PATH)
df["REC_ID"] = df["REC_ID"].astype(str).str.strip().str.lower()

rec_ids, lats, lons = [], [], []
for _, row in df.iterrows():
    geometry_str = str(row["geometry"])
    try:
        geometry = json.loads(geometry_str.replace("'", '"'))
        coords = geometry.get("coordinates", [])
        if coords and isinstance(coords[0], list) and len(coords[0]) == 2:
            lon, lat = coords[0]
            rec_ids.append(row["REC_ID"]); lats.append(lat); lons.append(lon)
        elif isinstance(coords, (list, tuple)) and len(coords) == 2:
            lon, lat = coords
            rec_ids.append(row["REC_ID"]); lats.append(lat); lons.append(lon)
    except Exception:
        pass

sites = pd.DataFrame({"REC_ID": rec_ids, "lat": lats, "lon": lons}).dropna().drop_duplicates("REC_ID")


def load_gcc_timeseries(pkl_dir: str, which: str = "Tree") -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for p in sorted(Path(pkl_dir).glob("loc*.pkl")):
        rec_id = p.stem.lower().replace("loc", "")
        with open(p, "rb") as f:
            results = pickle.load(f)

        ts = results.get(which, {}).get("time_series", pd.DataFrame())
        if isinstance(ts, pd.DataFrame) and not ts.empty:
            s = (ts.loc[:, ["datetime", "value"]]
                   .assign(datetime=lambda d: pd.to_datetime(d["datetime"], errors="coerce"))
                   .dropna()
                   .groupby(pd.Grouper(key="datetime", freq="D"))["value"].mean()
                   .sort_index())
            if not s.empty:
                out[rec_id] = s
    return out

series_by_site = load_gcc_timeseries(PKL_DIR, which="Tree")



sites = sites[sites["REC_ID"].isin(series_by_site.keys())].reset_index(drop=True)
if sites.empty:
    raise SystemExit("No overlap between sites and time series.")

overlap_ids = set(sites["REC_ID"])
min_date = min(s.index.min() for rid, s in series_by_site.items() if rid in overlap_ids)
max_date = max(s.index.max() for rid, s in series_by_site.items() if rid in overlap_ids)
dates = pd.date_range(min_date, max_date, freq="D")

FILL_LIMIT = 7 
series_ff = {}
for rec_id in sites["REC_ID"]:
    s = series_by_site.get(rec_id)
    if s is None or s.empty:
        continue
    # make sure daily & dedup by day
    s = s.copy()
    s.index = pd.to_datetime(s.index).floor("D")
    s = s.groupby(level=0).mean()
    series_ff[rec_id] = s.reindex(dates).ffill(limit=FILL_LIMIT)

sites = sites[sites["REC_ID"].isin(series_ff.keys())].reset_index(drop=True)
if sites.empty:
    raise SystemExit("No sites left after aligning/ffill.")

all_vals = pd.concat(series_ff.values()).astype(float).to_numpy()
all_vals = all_vals[np.isfinite(all_vals)]
if all_vals.size:
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
else:
    vmin, vmax = 0.0, 1.0
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
    vmin, vmax = 0.0, 1.0

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import PillowWriter

cmap = LinearSegmentedColormap.from_list("w2g", ["#FFFFFF", "#006400"])

# grid size (parse into (50*50) in this case)
NX, NY = 50, 50

xmin, xmax = float(sites["lon"].min()), float(sites["lon"].max())
ymin, ymax = float(sites["lat"].min()), float(sites["lat"].max())
pad_x = max(0.01, (xmax - xmin) * 0.05) if np.isfinite(xmin + xmax) else 0.01
pad_y = max(0.01, (ymax - ymin) * 0.05) if np.isfinite(ymin + ymax) else 0.01
xmin -= pad_x; xmax += pad_x
ymin -= pad_y; ymax += pad_y

xg = np.linspace(xmin, xmax, NX)
yg = np.linspace(ymin, ymax, NY)
Xg, Yg = np.meshgrid(xg, yg)

P = sites[["lon","lat"]].to_numpy()
tree = cKDTree(P)
dists, _ = tree.query(P, k=4)   # k=1 is self (0), so k=4 gives 3rd NN
nn3 = dists[:, 3]
epsilon = float(np.nanmedian(nn3))
if not np.isfinite(epsilon) or epsilon <= 0:
    # fallback: a fraction of the domain size
    epsilon = 0.04 * max((sites["lon"].max()-sites["lon"].min()),
                         (sites["lat"].max()-sites["lat"].min()))

SMOOTH_FRAC = 0.02
value_span = max(vmax - vmin, 1e-6)
smooth = SMOOTH_FRAC * value_span


fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("Lon"); ax.set_ylabel("Lat")

cmap = LinearSegmentedColormap.from_list("w2g", ["#FFFFFF", "#006400"])
cmap.set_bad(alpha=0.0)

Z0 = np.full((NY, NX), np.nan, dtype=float)
im = ax.imshow(
    Z0, origin="lower", extent=(xmin, xmax, ymin, ymax),
    cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.95, interpolation="nearest"
)

vals0 = np.full(len(sites), vmin, dtype=float)  # start as white
sc = ax.scatter(
    sites["lon"], sites["lat"],
    c=vals0, s=50, cmap=cmap, vmin=vmin, vmax=vmax,
    edgecolor="k", linewidths=0.2
)
cax = inset_axes(
    ax,
    width="3%",
    height="100%",
    loc="lower left",
    bbox_to_anchor=(1.01, 0., 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
cb = fig.colorbar(im, cax=cax)
cb.set_label("GCC")

cb.set_label("GCC")
ax.set_title("GCC by Site")


# Use absolute dates
FRAME_START = pd.Timestamp("2024-01-01")
FRAME_END   = pd.Timestamp("2024-12-31")
frame_dates = dates[(dates >= FRAME_START) & (dates <= FRAME_END)]

SCALE_TO_WINDOW = True
if SCALE_TO_WINDOW:
    all_vals_window = pd.concat([series_ff[rid].reindex(frame_dates)
                                 for rid in sites["REC_ID"]]).astype(float).to_numpy()
    all_vals_window = all_vals_window[np.isfinite(all_vals_window)]
    if all_vals_window.size:
        vmin, vmax = float(np.nanmin(all_vals_window)), float(np.nanmax(all_vals_window))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        # update clim if 'im' and 'sc' already exist
        try:
            im.set_clim(vmin=vmin, vmax=vmax)
            sc.set_clim(vmin=vmin, vmax=vmax)
            cb.update_normal(im)
        except Exception:
            pass

# Toggle interpolator: "grid" for griddata, "rbf" for RBF
INTERP = "grid"  # or "rbf"

frames_written = 0
writer = PillowWriter(fps=24)

with writer.saving(fig, OUTPUT_GIF, dpi=120):
    for d in frame_dates:
        # Pull GCC in same order as 'sites'
        vals = np.array(
            [series_ff.get(rec_id, pd.Series(dtype=float)).get(d, np.nan)
             for rec_id in sites["REC_ID"]],
            dtype=float
        )
        mask = np.isfinite(vals)

        # Compute continuous field Z on (Xg, Yg)
        try:
            if mask.sum() >= 3:
                if INTERP == "rbf":
                    # RBF params should be precomputed above (epsilon, smooth); fall back if missing
                    try:
                        rbf = Rbf(P[mask, 0], P[mask, 1], vals[mask],
                                  function="multiquadric",
                                  epsilon=epsilon if 'epsilon' in globals() else None,
                                  smooth=smooth if 'smooth' in globals() else 0.0)
                        Z = rbf(Xg, Yg)
                    except Exception:
                        Zlin = griddata(P[mask], vals[mask], (Xg, Yg), method="linear")
                        Znear = griddata(P[mask], vals[mask], (Xg, Yg), method="nearest")
                        Z = np.where(np.isnan(Zlin), Znear, Zlin)
                else:
                    Zlin = griddata(P[mask], vals[mask], (Xg, Yg), method="linear")
                    Znear = griddata(P[mask], vals[mask], (Xg, Yg), method="nearest")
                    Z = np.where(np.isnan(Zlin), Znear, Zlin)
            else:
                # Too few finite points → blank field (transparent via cmap.set_bad)
                Z = np.full((NY, NX), np.nan, dtype=float)
        except Exception as e:
            print(f"[{d.date()}] interpolation failed: {e}")
            Z = np.full((NY, NX), np.nan, dtype=float)

        # Update plot (show missing dots as white = vmin so they’re visible)
        im.set_data(Z)
        sc.set_array(np.where(mask, vals, vmin))
        ax.set_title(f"GCC field — {pd.to_datetime(d).date()}")

        # Write frame
        writer.grab_frame()
        frames_written += 1

if frames_written == 0:
    raise RuntimeError(
        "No frames were written. Likely the time window had zero dates, or "
        "your data selection produced only NaNs. Loosen the window or reduce FILL_LIMIT."
    )

plt.close(fig)
print(f"Saved GIF: {OUTPUT_GIF} ({frames_written} frames)")
