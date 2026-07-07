import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lingam import DirectLiNGAM, ICALiNGAM
from sklearn.exceptions import ConvergenceWarning
from utils import DAGMA, gen_laplace, gen_t

from otlingam import ExhaustiveOTLiNGAM, GreedyOTLiNGAM, OTICALiNGAM, disorder

# Set plot parameters
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)

# Filter warnings from FastICA not converging
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set defaults
np.random.seed(42)
ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "Exhaustive OT-LiNGAM": ExhaustiveOTLiNGAM,
    "Greedy OT-LiNGAM": GreedyOTLiNGAM,
    "OT-ICA-LiNGAM": OTICALiNGAM,
    "ICA-LiNGAM": ICALiNGAM,
    "Direct-LiNGAM": DirectLiNGAM,
    "DAGMA": DAGMA,
}
GRAPH_CONFIGURATIONS = (("er", 2), ("er", 4), ("sf", 2), ("sf", 4))
N_RUNS = 20
N_RANGE = (100, 250, 500, 1000, 1500)
D_RANGE = (4, 6, 8, 10, 12)
FIXED_N = 1000
FIXED_D = 7
HETEROGENEITY_N = 3000
HETEROGENEITY_D = 8
MIN_DF = 2.5
MAX_DF_RANGE = (2.5, 5, 10, 20, 40)


def nd_results(graph_type, edges_per_node):
    results = []
    for sweep, grid, fixed_n, fixed_d in (
        ("n", N_RANGE, None, FIXED_D),
        ("d", D_RANGE, FIXED_N, None),
    ):
        for value in grid:
            for run in range(N_RUNS):
                data, weights = gen_laplace(
                    fixed_n or value,
                    fixed_d or value,
                    edges_per_node,
                    graph_type=graph_type,
                )
                for name, factory in MODELS.items():
                    model = (
                        factory(random_state=run)
                        if "ICA" in name or "Direct" in name
                        else factory()
                    )
                    model.fit(data)
                    results.append(
                        {
                            "Sweep": sweep,
                            "Value": value,
                            "Method": name,
                            "Disorder": disorder(model.causal_order_, weights),
                        }
                    )

    return pd.DataFrame(results)


def heterogeneity_results(graph_type, edges_per_node):
    results = []
    for maximum_df in MAX_DF_RANGE:
        for run in range(N_RUNS):
            data, weights = gen_t(
                HETEROGENEITY_N,
                HETEROGENEITY_D,
                edges_per_node,
                np.linspace(MIN_DF, maximum_df, HETEROGENEITY_D),
                graph_type=graph_type,
            )
            for name, factory in MODELS.items():
                model = (
                    factory(random_state=run)
                    if "ICA" in name or "Direct" in name
                    else factory()
                )
                model.fit(data)
                results.append(
                    {
                        "Value": maximum_df,
                        "Method": name,
                        "Disorder": disorder(model.causal_order_, weights),
                    }
                )

    return pd.DataFrame(results)


def plot(axis, results, xlabel, title, legend):
    sns.lineplot(
        data=results,
        x="Value",
        y="Disorder",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax=axis,
        legend=legend,
    )
    axis.set(xlabel=xlabel, ylabel="Disorder", title=title)
    if legend:
        axis.legend(loc="upper left")
    axis.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--nd", action="store_true")
    mode.add_argument("--heterogeneity", action="store_true")
    args = parser.parse_args()

    if args.nd:
        figure, axes = plt.subplots(2, 4, figsize=(28, 10), layout="constrained")
        figure.suptitle("Disorder with varying sample size and dimension")
        for column, (graph_type, edges_per_node) in enumerate(GRAPH_CONFIGURATIONS):
            results = nd_results(graph_type, edges_per_node)
            graph_title = f"{graph_type}{edges_per_node}".upper()
            for row, (sweep, xlabel) in enumerate(
                (
                    ("n", f"n (sample size), d = {FIXED_D}"),
                    ("d", f"d (dimension), n = {FIXED_N}"),
                )
            ):
                plot(
                    axes[row, column],
                    results[results["Sweep"] == sweep],
                    xlabel,
                    graph_title,
                    row == 0 and column == 0,
                )
        output = ROOT / "figures" / "varying-nd-disorder.pdf"
    else:
        figure, axes = plt.subplots(1, 4, figsize=(28, 5), layout="constrained")
        figure.suptitle("Disorder under noise heterogeneity")
        for column, (graph_type, edges_per_node) in enumerate(GRAPH_CONFIGURATIONS):
            results = heterogeneity_results(graph_type, edges_per_node)
            graph_title = f"{graph_type}{edges_per_node}".upper()
            plot(
                axes[column],
                results,
                "Maximum degrees of freedom",
                graph_title,
                column == 0,
            )
        output = ROOT / "figures" / "noise-heterogeneity-disorder.pdf"

    figure.savefig(output)
    plt.show()


if __name__ == "__main__":
    main()
