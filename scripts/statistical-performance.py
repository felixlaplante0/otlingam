import argparse
import warnings

import lingam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score

from otlingam import ExhaustiveLiNGAM, GreedyLiNGAM, ICALiNGAM, disorder
from utils import gen_laplace, gen_t

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
warnings.filterwarnings(
    "ignore", message="FastICA did not converge.*", category=ConvergenceWarning
)

# Set defaults
MODELS = {
    "ExhaustiveLiNGAM": ExhaustiveLiNGAM,
    "GreedyLiNGAM": GreedyLiNGAM,
    "OT ICA-LiNGAM": ICALiNGAM,
    "ICA-LiNGAM": lingam.ICALiNGAM,
    "DirectLiNGAM": lingam.DirectLiNGAM,
}
METRICS = ("Disorder", "Structural Hamming distance", "F1 score")
rng = np.random.default_rng(0)


def fit(data, weights, seed):
    for name, factory in MODELS.items():
        model = (
            factory(random_state=seed)
            if "ICA" in name or "Direct" in name
            else factory()
        )
        model.fit(data)
        estimated = np.abs(model.adjacency_matrix_) > 0.1
        truth = weights != 0
        yield {
            "Method": name,
            "Disorder": disorder(model.causal_order_, weights),
            "Structural Hamming distance": np.count_nonzero(truth != estimated),
            "F1 score": f1_score(truth.ravel(), estimated.ravel()),
        }


def nd_results():
    results = []
    for sweep, grid, fixed_n, fixed_d in (
        ("n", (250, 500, 1000, 2000, 4000), None, 7),
        ("d", (4, 6, 8, 10), 1000, None),
    ):
        for value in grid:
            for repetition in range(10):
                data, weights = gen_laplace(
                    fixed_n or value, fixed_d or value, 0.4, rng
                )
                results.extend(
                    {"Sweep": sweep, "Value": value, **result}
                    for result in fit(data, weights, repetition)
                )

    return pd.DataFrame(results)


def heterogeneity_results():
    results = []
    for maximum_df in (3, 5, 10, 20, 40):
        for repetition in range(12):
            data, weights = gen_t(3000, 8, 0.4, np.linspace(3, maximum_df, 8), rng)
            results.extend(
                {"Value": maximum_df, **result}
                for result in fit(data, weights, repetition)
            )

    return pd.DataFrame(results)


def plot_row(axes, results, xlabel):
    for axis, metric in zip(axes, METRICS):
        sns.lineplot(
            data=results,
            x="Value",
            y=metric,
            hue="Method",
            marker="o",
            errorbar="sd",
            ax=axis,
            legend=metric == "Disorder",
        )
        axis.set(xlabel=xlabel, ylabel=metric, title=metric)
        if metric == "Disorder":
            axis.legend(loc="upper left")
        axis.grid(alpha=0.3)


parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--nd", action="store_true")
mode.add_argument("--heterogeneity", action="store_true")
args = parser.parse_args()

if args.nd:
    results = nd_results()
    figure, axes = plt.subplots(2, 3, figsize=(24, 10), layout="constrained")
    for row, (sweep, xlabel) in enumerate(
        (("n", "n (sample size), d = 7"), ("d", "d (dimension), n = 1000"))
    ):
        plot_row(axes[row], results[results["Sweep"] == sweep], xlabel)
    output = "../figures/varying-nd.pdf"
else:
    results = heterogeneity_results()
    figure, axes = plt.subplots(1, 3, figsize=(24, 5), layout="constrained")
    figure.suptitle("Noise heterogeneity")
    plot_row(axes, results, "Maximum degrees of freedom")
    output = "../figures/noise-heterogeneity.pdf"

figure.savefig(output)
plt.show()
