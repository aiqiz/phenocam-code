import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def load_loc_jsons(loc_dir: str | Path):
    loc_dir = Path(loc_dir)
    files = sorted(loc_dir.glob(f"{loc_dir.name}_*.json"))  # e.g. loc8087_202308.json
    rows = []
    for jf in files:
        with open(jf, "r", encoding="utf-8") as f:
            recs = json.load(f)
        rows.extend(recs)
    if not rows:
        raise FileNotFoundError(f"No JSON records found under {loc_dir}")
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["grass_mean"] = df["GCC_grass"].apply(lambda d: d["mean"])
    df["grass_std"]  = df["GCC_grass"].apply(lambda d: d["std"])
    df["tree_mean"]  = df["GCC_tree"].apply(lambda d: d["mean"])
    df["tree_std"]   = df["GCC_tree"].apply(lambda d: d["std"])
    return df.sort_values("datetime")

def plot_gcc_multi_month(loc_dir: str | Path, title_suffix: str = "", save_png: str | None = None):
    df = load_loc_jsons(loc_dir)
    grass = df.dropna(subset=["grass_mean"])
    tree  = df.dropna(subset=["tree_mean"])

    plt.figure(figsize=(8,6))

    # Grass
    grass["grass_mean"] = lowess(grass["grass_mean"], grass["datetime"], frac=0.03, return_sorted=False)
    plt.plot(grass["datetime"], grass["grass_mean"], label="Grass GCC", color="limegreen")
    plt.fill_between(grass["datetime"],
                     grass["grass_mean"] - grass["grass_std"],
                     grass["grass_mean"] + grass["grass_std"],
                     color="limegreen", alpha=0.2, label="±std")

    # Tree
    tree["tree_mean"] = lowess(tree["tree_mean"], tree["datetime"], frac=0.03, return_sorted=False)
    plt.plot(tree["datetime"], tree["tree_mean"], label="Tree GCC", color="forestgreen")
    plt.fill_between(tree["datetime"],
                     tree["tree_mean"] - tree["tree_std"],
                     tree["tree_mean"] + tree["tree_std"],
                     color="forestgreen", alpha=0.2, label="±std")


    plt.title(f"GCC over time - {Path(loc_dir).name} {title_suffix}".strip())
    plt.xlabel("Datetime")
    plt.ylabel("GCC")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_png:
        out = Path(save_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=180)
        print(f"Saved plot → {out}")
    else:
        plt.show()



plot_gcc_multi_month("../loc8077",
                     title_suffix="(2023)")
