import time

import lingam
import numpy as np
import pandas as pd

from exactdag import ExactDAG


def generate_data(n, d, graph_type="er", p=0.3):
    B = np.zeros((d, d))
    for i in range(1, d):
        if graph_type == "er":
            idx = [j for j in range(i) if np.random.rand() < p]
        else:
            degrees = np.sum(B[:i, :i] != 0, axis=0) + np.sum(B[:i, :i] != 0, axis=1)
            prob = np.ones(i) / i if np.sum(degrees) == 0 else degrees / np.sum(degrees)
            idx = np.random.choice(i, size=min(i, 2), replace=False, p=prob)
        for j in idx:
            B[i, j] = np.random.uniform(0.5, 2.0)

    p = np.random.permutation(d)
    B = B[np.ix_(p, p)]
    X = np.random.uniform(-1, 1, (n, d)) @ np.linalg.inv(np.eye(d) - B).T

    return X, B


def run(n=500, d=10, n_runs=10, graph_type="er", penalty=1e-3):
    res = []
    for _ in range(n_runs):
        X, B_true = generate_data(n, d, graph_type)
        for name, model in {
            "ExactDAG": ExactDAG(penalty=penalty),
            "ICA": lingam.ICALiNGAM(),
            "Direct": lingam.DirectLiNGAM(),
        }.items():
            t0 = time.perf_counter()
            model.fit(X)
            dt = time.perf_counter() - t0
            shd = np.sum((B_true != 0) != (model.adjacency_matrix_ != 0))
            res.append(
                {
                    "Method": name,
                    "SHD": shd,
                    "Edges": np.sum(model.adjacency_matrix_ != 0),
                    "Time": dt,
                }
            )

    df = pd.DataFrame(res).groupby("Method").agg(["mean", "std"])
    out = pd.DataFrame()
    for col in ["SHD", "Edges", "Time"]:
        out[col] = df[col].apply(
            lambda x: (
                f"{x['mean']:.2f} ± {x['std']:.2f}"
                if col != "Time"
                else f"{x['mean']:.3f}s"
            ),
            axis=1,
        )
    print(f"\n--- {graph_type.upper()} (n={n}, d={d}, runs={n_runs}) ---\n{out}")


if __name__ == "__main__":
    np.random.seed(42)
    for g in ["er", "sf"]:
        run(n=1000, d=10, n_runs=10, graph_type=g, penalty=1e-3)
