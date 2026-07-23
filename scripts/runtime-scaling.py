import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lingam import DirectLiNGAM, ICALiNGAM
from otlingam import ExhaustiveOTLiNGAM, GreedyOTLiNGAM, OTICALiNGAM

from _utils import DAGMA, gen_laplace

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
ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "Exhaustive OT-LiNGAM": ExhaustiveOTLiNGAM,
    "Greedy LO-LiNGAM": GreedyOTLiNGAM,
    "OT-ICA-LiNGAM": OTICALiNGAM,
    "ICA-LiNGAM": ICALiNGAM,
    "Direct-LiNGAM": DirectLiNGAM,
    "DAGMA": DAGMA,
}
N_RUNS = 10
N_RANGE = (250, 500, 1000, 2000, 4000)
D_RANGE = (6, 8, 10, 12, 16, 20)
FIXED_N = 1000
FIXED_D = 8
EDGES_PER_NODE = 2
GRAPH_TYPE = "er"


def main():
    # Warmup run to avoid including compilation time in the timing results
    warmup_data, _ = gen_laplace(
        FIXED_N,
        FIXED_D,
        EDGES_PER_NODE,
        graph_type=GRAPH_TYPE,
    )
    ExhaustiveOTLiNGAM().fit(warmup_data)

    results = []
    for sweep, grid, fixed_n, fixed_d in (
        ("n", N_RANGE, None, FIXED_D),
        ("d", D_RANGE, FIXED_N, None),
    ):
        for value in grid:
            for run in range(N_RUNS):
                data, _ = gen_laplace(
                    fixed_n or value,
                    fixed_d or value,
                    EDGES_PER_NODE,
                    graph_type=GRAPH_TYPE,
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
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for axis, (sweep, xlabel, title) in zip(
        axes,
        (
            (
                "n",
                f"n (sample size), d = {FIXED_D}",
                f"{GRAPH_TYPE.upper()}{EDGES_PER_NODE} runtime scaling with sample size",
            ),
            (
                "d",
                f"d (dimension), n = {FIXED_N}",
                f"{GRAPH_TYPE.upper()}{EDGES_PER_NODE} runtime scaling with dimension",
            ),
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
            legend=axis is axes[0],
        )
        axis.set(xlabel=xlabel, ylabel="Runtime (seconds) ↓", title=title)
        if axis is axes[0]:
            axis.legend(loc="upper left")
        axis.set_yscale("log")
        axis.grid(alpha=0.3)

    (ROOT / "figures").mkdir(exist_ok=True)
    figure.savefig(ROOT / "figures" / "runtime-scaling.pdf")
    plt.show()


if __name__ == "__main__":
    main()
