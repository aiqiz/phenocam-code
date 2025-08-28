import json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# loaiding + normalization
def load_loc_jsons(loc_dir: Path) -> pd.DataFrame:
    files = sorted(loc_dir.glob(f"{loc_dir.name}_*.json"))
    rows = []
    for jf in files:
        with open(jf, "r", encoding="utf-8") as f:
            rows.extend(json.load(f))
    if not rows:
        return pd.DataFrame(columns=["datetime","grass_mean","tree_mean"])
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df["grass_mean"] = df["GCC_grass"].apply(lambda d: np.nan if d is None else d.get("mean"))
    df["tree_mean"]  = df["GCC_tree"].apply(lambda d: np.nan if d is None else d.get("mean"))
    return df[["datetime", "grass_mean", "tree_mean"]]


def normalize_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    vals = out[col].to_numpy(dtype=float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        out[col] = (vals - vmin) / (vmax - vmin)
    else:
        out[col] = 0.0
    return out


def make_series(df: pd.DataFrame, col: str, window: str = "7D", normalize: bool = True) -> pd.DataFrame:
    s = df[["datetime", col]].dropna().copy()
    if s.empty:
        return pd.DataFrame(columns=["datetime","value"])
    s = (
        s.set_index("datetime")[col]
          .rolling(window, min_periods=1).mean()
          .reset_index()
          .rename(columns={col: "value"})
    )
    if normalize:
        s = normalize_series(s, col="value")
    return s.sort_values("datetime")



# asymmetric double sigmoid
def ads_model(t, w1, t1, w2, t2, b, m):
    return b + m * (
        (1 / (1 + np.exp(-w1 * (t - t1))))
        * (1 - (1 / (1 + np.exp(-w2 * (t - t2)))))
    )


def fit_gcc_series(dates, values):
    doy = pd.to_datetime(dates).dt.dayofyear.to_numpy(dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(doy) & np.isfinite(y)
    t, y = doy[mask], y[mask]
    if len(t) < 6:
        raise ValueError("Not enough points to fit")
    p0 = [0.2, 130, 0.2, 280, 0.0, 0.5]
    bounds = ([0.01, 90, 0.01, 220, 0.0, 0.0],
              [2.0, 160, 2.0, 330, 1.0, 1.0])

    popt, _ = curve_fit(ads_model, t, y, p0=p0, bounds=bounds, maxfev=20000)
    return popt


def extract_season_dates(popt, year: int):
    t_range = np.linspace(1, 365, 365)
    y_fit = ads_model(t_range, *popt)
    b, m = popt[4], popt[5]
    peak_idx = int(np.argmax(y_fit))
    thr10, thr90 = b + 0.3*m, b + 0.3*m
    sos, eos = None, None
    for t, y in zip(t_range[:peak_idx], y_fit[:peak_idx]):
        if y > thr10:
            sos = t; break
    for t, y in zip(t_range[peak_idx:], y_fit[peak_idx:]):
        if y < thr90:
            eos = t; break
    los = eos - sos if (sos is not None and eos is not None) else None
    peak_time, peak_value = float(t_range[peak_idx]), float(y_fit[peak_idx])

    def doy_to_date(doy):
        return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(doy)-1)

    return {
        "SOS": None if sos is None else doy_to_date(sos),
        "EOS": None if eos is None else doy_to_date(eos),
        "LOS": los,
        "PeakDate": doy_to_date(peak_time),
        "PeakValue": peak_value
    }


def evaluate_ads_on_dates(popt, dates: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Evaluate ADS on a date grid (uses DOY)."""
    t_doy = pd.to_datetime(dates).dayofyear.to_numpy(dtype=float)
    return ads_model(t_doy, *popt)


# per location plotting (saved)
def plot_location(loc_name: str, results: dict, out_plot_dir: Path, window: str, step: int):
    """Make a 2-panel plot (Grass & Tree) showing smoothed series, discrete fit points, and ADS fit."""
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {"Grass": "limegreen", "Tree": "forestgreen"}

    for ax, (name, payload) in zip(axes, results.items()):
        ts = payload.get("time_series", pd.DataFrame())
        popt = payload.get("params", None)
        metrics = payload.get("metrics", {})

        if ts.empty:
            ax.set_title(f"{name}: no data"); ax.axis("off"); continue

        # smoothed series
        ax.plot(ts["datetime"], ts["value"], color=colors[name], alpha=0.6, label=f"{name} ({window} mean, norm)")

        # discrete fit points
        ts_fit = ts.iloc[::step]
        ax.plot(ts_fit["datetime"], ts_fit["value"], "o", ms=4, mfc="white", mec=colors[name], label="fit points")

        # ADS fit
        if popt is not None:
            grid = pd.date_range(ts["datetime"].min(), ts["datetime"].max(), freq="D")
            yhat = evaluate_ads_on_dates(popt, grid)
            ax.plot(grid, yhat, color=colors[name], lw=2, label="ADS fit")

            # mark SOS/EOS/Peak
            sos, eos, peak = metrics.get("SOS"), metrics.get("EOS"), metrics.get("PeakDate")
            for when, style, lab in [(sos, "--", "SOS"), (eos, "--", "EOS"), (peak, ":" , "Peak")]:
                if when is not None:
                    ax.axvline(when, color=colors[name], linestyle=style, alpha=0.5, label=lab)

        ax.set_ylabel("Normalized GCC")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(loc="upper left")
        ax.set_title(f"{loc_name} — {name}")

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    out_png = out_plot_dir / f"{loc_name}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved plot → {out_png}")


def batch_process(root_dir: str, out_dir: str, window="7D", step=7):
    root = Path(root_dir)
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    plot_dir = outp / "plots"

    for loc_dir in sorted(root.glob("loc*")):
        df = load_loc_jsons(loc_dir)
        if df.empty:
            continue

        results = {}
        for name, col in [("Grass","grass_mean"), ("Tree","tree_mean")]:
            ts = make_series(df, col, window, normalize=True)
            if ts.empty:
                results[name] = {"time_series": ts, "params": None, "metrics": {}}
                continue

            # subsample discrete points
            ts_fit = ts.iloc[::step]
            try:
                popt = fit_gcc_series(ts_fit["datetime"], ts_fit["value"])
                year = ts["datetime"].dt.year.mode()[0]  # most common year
                metrics = extract_season_dates(popt, year)
            except Exception as e:
                print(f"[{loc_dir.name}] {name} fit failed: {e}")
                popt, metrics = None, {}

            results[name] = {"time_series": ts, "params": popt, "metrics": metrics}

        # save results
        out_file = outp / f"{loc_dir.name}.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved {loc_dir.name} → {out_file}")

        # plot per location
        plot_location(loc_dir.name, results, plot_dir, window, step)


batch_process(
    "/Users/aiqizhang/Desktop/traffic/gcc_2023",   # locXXXX folders
    "/Users/aiqizhang/Desktop/traffic/map_prepare_2023",   # outputs (pkl + plots/)
    window="3D", step=1
)
