# 📊 Optimal Transport LiNGAM

**otlingam** is a Python package for causal discovery in linear non-Gaussian structural equation models. It learns causal orders by maximizing the Wasserstein non-Gaussianity of standardized regression residuals and estimates edge weights with adaptive lasso.

---

## ✨ Features

- **Exhaustive causal-order learning**: `ExhaustiveOTLiNGAM` uses subset dynamic programming to find a globally optimal order.
- **Scalable greedy learning**: `GreedyOTLiNGAM` constructs an order by sequentially selecting the most non-Gaussian residual.
- **Optimal transport ICA**: `OTICALiNGAM` uses `OTICA` with FastICA initialization in the classical ICA-LiNGAM pipeline.
- **Exact empirical criterion**: Computes one-dimensional Wasserstein scores directly from ordered residuals and Gaussian quantiles.
- **LiNGAM integration**: Exposes causal orders and weighted adjacency matrices through the established LiNGAM estimator API.

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
pip install otlingam
```

## 🔧 Usage

### Example

The following example simulates a linear non-Gaussian structural equation model, learns a causal order with `GreedyOTLiNGAM`, and compares the true and estimated weighted adjacency matrices.

```python
import matplotlib.pyplot as plt
import numpy as np
from otlingam import GreedyOTLiNGAM, disorder

rng = np.random.default_rng(42)
n_samples = 5_000
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

`ExhaustiveOTLiNGAM` provides global order optimization at an exponential cost in the number of variables. `GreedyOTLiNGAM` provides a quadratic-time alternative. Set `fit_intercept=False` when the observations are already centered. The default `fit_intercept=True` centers the data and exposes the fitted intercepts through `intercept_`.

---

## 📖 Learn More

For configuration details and the API reference, visit [otlingam's documentation](https://felixlaplante0.github.io/otlingam).
