import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lingam import DirectLiNGAM, ICALiNGAM
from utils import DAGMA, gen_laplace

from otlingam import ExhaustiveOTLiNGAM, GreedyOTLiNGAM, OTICALiNGAM

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

# Set defaults
np.random.seed(42)
MODELS = {
    "Exhaustive OT-LiNGAM": ExhaustiveOTLiNGAM,
    "Greedy LO-LiNGAM": GreedyOTLiNGAM,
    "OT-ICA-LiNGAM": OTICALiNGAM,
    "ICA-LiNGAM": ICALiNGAM,
    "Direct-LiNGAM": DirectLiNGAM,
    "DAGMA": DAGMA,
}
N_RUNS = 10

# Warmup run to avoid including compilation time in the timing results
warmup_data, _ = gen_laplace(250, 7, 2, graph_type="er")
ExhaustiveOTLiNGAM().fit(warmup_data)

results = []
for sweep, grid, fixed_n, fixed_d in (
    ("n", (250, 500, 1000, 2000, 4000), None, 7),
    ("d", (6, 8, 10, 12, 16, 20), 1000, None),
):
    for value in grid:
        for run in range(N_RUNS):
            data, _ = gen_laplace(
                fixed_n or value,
                fixed_d or value,
                2,
                graph_type="er",
            )
            for name, factory in MODELS.items():
                model = (
                    factory(random_state=run)
                    if "ICA" in name or "Direct" in name
                    else factory()
                )
                start = time.perf_counter()
                model.fit(data)
                results.append(
                    {
                        "Sweep": sweep,
                        "Value": value,
                        "Method": name,
                        "Runtime (seconds)": time.perf_counter() - start,
                    }
                )

results = pd.DataFrame(results)
figure, axes = plt.subplots(1, 2, figsize=(16, 5), layout="constrained")
for axis, (sweep, xlabel, title) in zip(
    axes,
    (
        ("n", "n (sample size), d = 7", "ER2 runtime scaling with sample size"),
        ("d", "d (dimension), n = 1000", "ER2 runtime scaling with dimension"),
    ),
):
    sns.lineplot(
        data=results[results["Sweep"] == sweep],
        x="Value",
        y="Runtime (seconds)",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax=axis,
    )
    axis.set(xlabel=xlabel, ylabel="Runtime (seconds)", title=title)
    axis.legend(loc="upper left")
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
figure.savefig("../figures/runtime-scaling.pdf")
plt.show()
