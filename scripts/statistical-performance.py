import argparse
import warnings

import lingam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning

from otlingam import ExhaustiveLiNGAM, GreedyLiNGAM, ICALiNGAM, disorder
from utils import DAGMA, gen_laplace, gen_t

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
MODELS = {
    "ExhaustiveLiNGAM": ExhaustiveLiNGAM,
    "GreedyLiNGAM": GreedyLiNGAM,
    "OT ICA-LiNGAM": ICALiNGAM,
    "ICA-LiNGAM": lingam.ICALiNGAM,
    "DirectLiNGAM": lingam.DirectLiNGAM,
    "DAGMA": DAGMA,
}
GRAPH_CONFIGURATIONS = (("er", 2), ("er", 4), ("sf", 2), ("sf", 4))
n_runs = 20
np.random.seed(42)


def nd_results(graph_type, edges_per_node):
    results = []
    for sweep, grid, fixed_n, fixed_d in (
        ("n", (100, 250, 500, 1000, 1500), None, 7),
        ("d", (4, 6, 8, 10, 12), 1000, None),
    ):
        for value in grid:
            for run in range(n_runs):
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
    for maximum_df in (2.5, 5, 10, 20, 40):
        for run in range(n_runs):
            data, weights = gen_t(
                3000,
                8,
                edges_per_node,
                np.linspace(2.5, maximum_df, 8),
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
            (("n", "n (sample size), d = 7"), ("d", "d (dimension), n = 1000"))
        ):
            plot(
                axes[row, column],
                results[results["Sweep"] == sweep],
                xlabel,
                graph_title,
                row == 0 and column == 0,
            )
    output = "../figures/varying-nd-disorder.pdf"
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
    output = "../figures/noise-heterogeneity-disorder.pdf"

figure.savefig(output)
plt.show()
