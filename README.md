<p align="center">
  <img src="https://raw.githubusercontent.com/felixlaplante0/otlingam/main/docs/source/_static/otlingam-logo.svg" alt="OTLiNGAM logo" width="128">
</p>

<h1 align="center">OT-LiNGAM</h1>

<p align="center"><strong>Optimal Transport LiNGAM.</strong><br>
Scikit-learn-compatible causal discovery for linear non-Gaussian systems.</p>

<p align="center">
  <a href="https://otlingam.readthedocs.io/en/latest/">Documentation</a> ·
  <a href="https://pypi.org/project/otlingam/">PyPI</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/otlingam/"><img src="https://img.shields.io/pypi/v/otlingam?logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://pypi.org/project/otlingam/"><img src="https://img.shields.io/badge/python-3.11--3.14-blue?logo=python&logoColor=white" alt="Supported Python versions: 3.11–3.14"></a>
  <a href="https://github.com/felixlaplante0/otlingam/actions/workflows/lint.yml"><img src="https://github.com/felixlaplante0/otlingam/actions/workflows/lint.yml/badge.svg" alt="Lint status"></a>
  <a href="https://codecov.io/gh/felixlaplante0/otlingam"><img src="https://codecov.io/gh/felixlaplante0/otlingam/graph/badge.svg" alt="Coverage"></a>
  <a href="https://otlingam.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/otlingam/badge/?version=latest" alt="Documentation status"></a>
  <a href="https://github.com/felixlaplante0/otlingam/blob/main/LICENSE"><img src="https://img.shields.io/github/license/felixlaplante0/otlingam" alt="License"></a>
</p>

**otlingam** is a Python package for causal discovery in linear non-Gaussian structural equation models. It learns causal orders by maximizing the Wasserstein non-Gaussianity of standardized regression residuals and estimates edge weights with adaptive Lasso.

---

## ✨ Features

- **Exhaustive causal-order learning**: ``ExhaustiveOTLiNGAM`` uses subset dynamic programming to find a globally optimal order.
- **Scalable greedy learning**: ``GreedyOTLiNGAM`` constructs an order by sequentially selecting the most non-Gaussian residual.
- **Optimal transport ICA**: ``OTICALiNGAM`` uses ``OTICA`` with FastICA initialization in the classical ICA-LiNGAM pipeline.
- **Exact empirical criterion**: Computes one-dimensional Wasserstein scores directly from ordered residuals and Gaussian quantiles.
- **LiNGAM integration**: Exposes causal orders and weighted adjacency matrices through the established LiNGAM estimator API.
- **scikit-learn integration**: Native ``BaseEstimator`` integration with familiar ``fit``, ``get_params``, ``set_params``, and ``clone`` support.

The exhaustive estimator uses [djbsort](https://salsa.debian.org/debian/djbsort)
for double-precision residual sorting and [Google Highway](https://github.com/google/highway)
for SIMD kernels. Both projects are tracked as Git submodules; the sorter
selects a portable fallback or the CPU's AVX2/NEON implementation at runtime.

---

## ⚡ Method

The estimators assume the linear structural equation model

$$
X_j = \sum_{k \in \mathrm{Pa}(j)} B_{jk} X_k + \varepsilon_j,
$$

where the graph is acyclic and the structural noises are mutually independent, centered, and have finite nonzero variances. Causal-order identification additionally requires at most one Gaussian structural noise.

For a candidate order $\sigma$, let $R_j(\sigma)$ be the population residual obtained by regressing $X_j$ on its predecessors under $\sigma$. The oracle Wasserstein order objective is

$$
G(\sigma) = \sum_{j = 1}^{d} \mathcal{W}_2\left( \mathrm{std}(R_j(\sigma)), \mathcal{N}(0, 1) \right)^2.
$$

Given $n$ observations, let $\widehat{R}_j^{(i)}(\sigma)$ be the ordinary least-squares residual for observation $i$. OTLiNGAM maximizes the empirical order objective

$$
\widehat{G}_n(\sigma) = \sum_{j = 1}^{d} \mathcal{W}_2\left( \mathrm{std}\left( \frac{1}{n} \sum_{i = 1}^{n} \delta_{\widehat{R}_j^{(i)}(\sigma)} \right), \mathcal{N}(0, 1) \right)^2.
$$

At the population level, the maximizers of $G$ are exactly the topological orders under the stated assumptions. A topological order exposes the independent structural noises as regression residuals, whereas an incorrect order may mix several noises and reduce the total objective. Each empirical one-dimensional Wasserstein distance is evaluated exactly by sorting the standardized residuals and comparing them with the Gaussian reference quantiles.

---

## 🚀 Installation

```bash
python -m pip install otlingam
```

## 🔧 Usage

### Example

The following example simulates a linear non-Gaussian structural equation model, learns a causal order with ``GreedyOTLiNGAM``, and compares the true and estimated weighted adjacency matrices.

```python
import matplotlib.pyplot as plt
import numpy as np
from otlingam import GreedyOTLiNGAM, disorder

rng = np.random.default_rng(42)
n_samples = 5000
adjacency_matrix = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.7, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.9, 0.0, 0.0],
        [0.0, -0.6, 0.0, 0.7, 0.0],
    ]
)
noise = rng.uniform(-1.0, 1.0, size=(n_samples, 5))
X = noise @ np.linalg.inv(np.eye(5) - adjacency_matrix).T

model = GreedyOTLiNGAM().fit(X)

print("Estimated causal order:", model.causal_order_)
print("Disorder:", disorder(model.causal_order_, adjacency_matrix))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
matrices = (adjacency_matrix, model.adjacency_matrix_)
titles = ("True adjacency matrix", "Estimated adjacency matrix")
for ax, matrix, title in zip(axes, matrices, titles, strict=True):
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Parent")
    ax.set_ylabel("Child")
fig.colorbar(image, ax=axes, label="Edge weight")

plt.show()
```

``ExhaustiveOTLiNGAM`` provides global order optimization at an exponential cost in the number of variables. ``GreedyOTLiNGAM`` provides a quadratic-time alternative. Set ``fit_intercept=False`` when the observations are already centered. The default ``fit_intercept=True`` centers the data and exposes the fitted intercepts through ``intercept_``.

---

## 📊 Reproducing Results

Clone the repository, create and activate a virtual environment, then install the exact package versions used for the paper:

```bash
python -m pip install -r scripts/requirements.txt
```

Keep the repository folder layout unchanged: do not move, rename, or flatten its folders, because the experiment scripts resolve paths relative to their file locations.

Run the experiment scripts from the repository root:

```bash
python scripts/statistical-performance.py --nd
python scripts/statistical-performance.py --heterogeneity
python scripts/statistical-performance.py --k
python scripts/runtime-scaling.py
```

Alternatively, run all experiments sequentially on Windows, Linux, or macOS:

```bash
python run-all.py
```

The runner uses the active Python interpreter and stops if an experiment fails.

These commands write `varying-nd-disorder.pdf`, `noise-heterogeneity-disorder.pdf`, `varying-k-performance.pdf`, and `runtime-scaling.pdf` to `figures/`. The paper settings (including random seed 42, sample sizes, dimensions, graph configurations, and run counts) are defined as constants near the top of each script.
