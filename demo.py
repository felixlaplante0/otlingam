import itertools
import time

import lingam
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from optidag import ExhaustiveDAG, GreedyDAG, disorder

NOISES = {
    "uniform": lambda size: np.random.uniform(-1, 1, size),
    "laplace": lambda size: np.random.laplace(size=size),
    "exponential": lambda size: np.random.exponential(size=size),
    "gumbel": lambda size: np.random.gumbel(size=size),
    "gamma": lambda size: np.random.gamma(2, size=size),
}

MODELS = {
    "ExhaustiveDAG": ExhaustiveDAG,
    "GreedyDAG": GreedyDAG,
    "ICA-LiNGAM": lingam.ICALiNGAM,
    "DirectLiNGAM": lingam.DirectLiNGAM,
}


def simulate_data(n, d, graph_type, noise_type):
    seed = int(np.random.randint(1 << 32))
    graph = (
        nx.gnp_random_graph(d, 0.3, seed=seed)
        if graph_type == "erdos-renyi"
        else nx.barabasi_albert_graph(d, 2, seed=seed)
    )
    adjacency = nx.to_numpy_array(graph)
    rank = np.argsort(np.random.permutation(d))
    weights = adjacency * (rank[:, None] > rank) * np.random.uniform(0.5, 2, (d, d))
    weights *= np.random.choice((-1, 1), (d, d))

    noise = NOISES[noise_type]((n, d))
    noise = (noise - noise.mean(0)) / noise.std(0)
    return np.linalg.solve(np.eye(d) - weights, noise.T).T, weights


def shd(true_weights, estimated_weights):
    return np.count_nonzero((true_weights != 0) != (estimated_weights != 0))


np.random.seed(42)
n = 1000
d = 10
repetitions = 10
results = []

for noise_type, graph_type, _ in itertools.product(
    NOISES, ("erdos-renyi", "scale-free"), range(repetitions)
):
    data, true_weights = simulate_data(n, d, graph_type, noise_type)

    for name, model_factory in MODELS.items():
        model = model_factory()
        start = time.perf_counter()
        model.fit(data)
        elapsed = time.perf_counter() - start

        results.append(
            {
                "Noise": noise_type,
                "Graph": graph_type,
                "Method": name,
                "SHD": shd(true_weights, model.adjacency_matrix_),
                "Disorder": disorder(model.causal_order_, true_weights),
                "Time": elapsed,
            }
        )

results = pd.DataFrame(results)
grouped = results.groupby(["Noise", "Graph", "Method"])
means = grouped.mean()
errors = grouped.std()

noise_types = list(NOISES)
graph_types = ("erdos-renyi", "scale-free")
x = np.arange(len(noise_types))
width = 0.2

fig, axes = plt.subplots(
    len(graph_types),
    3,
    figsize=(13, 7),
    sharex=True,
    layout="constrained",
)
metrics = (
    ("SHD", "Structural Hamming distance"),
    ("Disorder", "Disorder"),
    ("Time", "Fit time (seconds)"),
)

for row, graph_type in enumerate(graph_types):
    for column, (metric, title) in enumerate(metrics):
        axis = axes[row, column]
        for index, method in enumerate(MODELS):
            keys = [(noise, graph_type, method) for noise in noise_types]
            values = np.array([means.loc[key, metric] for key in keys])
            yerr = np.array([errors.loc[key, metric] for key in keys])
            axis.bar(
                x + (index - (len(MODELS) - 1) / 2) * width,
                values,
                width,
                yerr=yerr,
                capsize=2,
                label=method,
            )

        if row == 0:
            axis.set_title(title)
        axis.set_xticks(x, noise_types)
        axis.grid(axis="y", alpha=0.3)

    axes[row, 0].set_ylabel(f"{graph_type}\nMean")
    axes[row, 2].set_yscale("log")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="outside lower center",
    ncols=len(MODELS),
    frameon=False,
)
plt.show()
