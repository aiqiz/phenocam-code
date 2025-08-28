import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def readin():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Extract GCC values (may contain None)
    df["grass_mean"] = df["GCC_grass"].apply(lambda d: d["mean"])
    df["grass_std"]  = df["GCC_grass"].apply(lambda d: d["std"])
    df["tree_mean"]  = df["GCC_tree"].apply(lambda d: d["mean"])
    df["tree_std"]   = df["GCC_tree"].apply(lambda d: d["std"])

    # Drop rows with None
    df_grass = df.dropna(subset=["grass_mean"]).sort_values("datetime")
    df_tree  = df.dropna(subset=["tree_mean"]).sort_values("datetime")
    return df_grass, df_tree


def plot(df_grass, df_tree):
    plt.figure(figsize=(12,6))

    # Grass
    plt.plot(df_grass["datetime"], df_grass["grass_mean"],
            color="limegreen", label="Grass GCC")
    plt.fill_between(df_grass["datetime"],
                    df_grass["grass_mean"] - df_grass["grass_std"],
                    df_grass["grass_mean"] + df_grass["grass_std"],
                    color="limegreen", alpha=0.2)

    # Tree
    plt.plot(df_tree["datetime"], df_tree["tree_mean"],
            color="forestgreen", label="Tree GCC")
    plt.fill_between(df_tree["datetime"],
                    df_tree["tree_mean"] - df_tree["tree_std"],
                    df_tree["tree_mean"] + df_tree["tree_std"],
                    color="forestgreen", alpha=0.2)

    plt.title(f"GCC values over time – {JSON_FILE.stem}")
    plt.xlabel("Datetime")
    plt.ylabel("GCC")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


JSON_FILE = Path("../gcc_results_2023/loc8222/loc8222_202305.json")
df_grass, df_tree = readin()
plot(df_grass, df_tree)