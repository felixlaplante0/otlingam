Optimal Transport LiNGAM
========================

**otlingam** is a Python package for causal discovery in linear non-Gaussian structural equation models. It learns causal orders by maximizing the Wasserstein non-Gaussianity of standardized regression residuals and estimates edge weights with adaptive Lasso.

Features
--------

- **Exhaustive causal-order learning**: ``ExhaustiveOTLiNGAM`` uses subset dynamic programming to find a globally optimal order.
- **Scalable greedy learning**: ``GreedyOTLiNGAM`` constructs an order by sequentially selecting the most non-Gaussian residual.
- **Optimal transport ICA**: ``OTICALiNGAM`` uses ``OTICA`` with FastICA initialization in the classical ICA-LiNGAM pipeline.
- **Exact empirical criterion**: Computes one-dimensional Wasserstein scores directly from ordered residuals and Gaussian quantiles.
- **LiNGAM integration**: Exposes causal orders and weighted adjacency matrices through the established LiNGAM estimator API.

Method
------

The estimators assume the linear structural equation model

.. math::

   X_j = \sum_{k \in \operatorname{Pa}(j)} B_{jk} X_k + \varepsilon_j,

where the graph is acyclic and the structural noises are mutually independent, centered, and have finite nonzero variances. Causal-order identification additionally requires at most one Gaussian structural noise.

For a candidate order :math:`\sigma \in \mathfrak{S}_d`, let :math:`R_j(\sigma)` be the population residual obtained by regressing :math:`X_j` on its predecessors under :math:`\sigma`. The oracle Wasserstein order objective is

.. math::

   G(\sigma) = \sum_{j = 1}^{d} \mathcal{W}_2\left( \mathrm{std}\left( R_j(\sigma) \right), \mathcal{N}(0, 1) \right)^2.

Given observations :math:`X^{(1)}, \ldots, X^{(n)}`, let :math:`\widehat{R}_j^{(i)}(\sigma)` be the ordinary least-squares residual for observation :math:`i`. OTLiNGAM maximizes the empirical order objective

.. math::

   \widehat{\sigma}_n \in \operatorname*{\arg\max}_{\sigma \in \mathfrak{S}_d} \widehat{G}_n(\sigma) = \sum_{j = 1}^{d} \mathcal{W}_2\left( \mathrm{std}\left( \frac{1}{n} \sum_{i = 1}^{n} \delta_{\widehat{R}_j^{(i)}(\sigma)} \right), \mathcal{N}(0, 1) \right)^2.

At the population level, the maximizers of :math:`G` are exactly the topological orders under the stated assumptions. A topological order exposes the independent structural noises as regression residuals, whereas an incorrect order may mix several noises and reduce the total objective. Each empirical one-dimensional Wasserstein distance is evaluated exactly by sorting the standardized residuals and comparing them with the Gaussian reference quantiles.

Algorithms
----------

``ExhaustiveOTLiNGAM`` evaluates local residual scores and uses subset dynamic programming to recover a globally optimal order. It evaluates :math:`d 2^{d - 1}` local scores and stores :math:`O(2^d)` states, so its exponential dependence on :math:`d` limits it to smaller systems.

``GreedyOTLiNGAM`` repeatedly selects the most non-Gaussian standardized residual, removes its linear effect from the remaining variables, and continues on the residualized system. This avoids subset enumeration and provides a quadratic-time order procedure.

``OTICALiNGAM`` estimates an unmixing matrix with ``OTICA`` using FastICA initialization, then applies the standard ICA-LiNGAM permutation, scaling, and adjacency estimation steps.

Installation
------------

Install the package from PyPI:

.. code-block:: bash

   pip install otlingam

Usage
-----

The following example simulates a linear non-Gaussian structural equation model, learns a causal order with ``GreedyOTLiNGAM``, and compares the true and estimated weighted adjacency matrices.

.. code-block:: python

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

Configuration
-------------

``ExhaustiveOTLiNGAM`` provides global order optimization at an exponential cost in the number of variables. ``GreedyOTLiNGAM`` provides a quadratic-time alternative. Set ``fit_intercept=False`` when the observations are already centered. The default ``fit_intercept=True`` centers the data and exposes the fitted intercepts through ``intercept_``.

Fitted estimators expose ``causal_order_`` from source to sink, ``adjacency_matrix_`` with entry :math:`(j, k)` representing the effect :math:`k \to j`, ``score_`` for score-based estimators, and ``intercept_`` when intercept fitting is enabled.

API Reference
-------------

.. autoclass:: otlingam.ExhaustiveOTLiNGAM
   :members:
   :show-inheritance:

.. autoclass:: otlingam.GreedyOTLiNGAM
   :members:
   :show-inheritance:

.. autoclass:: otlingam.ICAiNGAM
   :members:
   :show-inheritance:

.. autofunction:: otlingam.disorder
