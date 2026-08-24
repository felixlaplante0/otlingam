Quick start
===========

Install OTLiNGAM with pip:

.. code-block:: bash

   python -m pip install otlingam

Generate data from a small linear non-Gaussian model and fit the scalable estimator:

.. code-block:: python

   import numpy as np
   from otlingam import GreedyOTLiNGAM

   rng = np.random.default_rng(42)
   adjacency = np.array(
       [
           [0.0, 0.0, 0.0, 0.0],
           [0.8, 0.0, 0.0, 0.0],
           [0.0, -0.7, 0.0, 0.0],
           [0.5, 0.0, 0.9, 0.0],
       ]
   )
   noise = rng.uniform(-1.0, 1.0, size=(2_000, 4))
   X = noise @ np.linalg.inv(np.eye(4) - adjacency).T

   model = GreedyOTLiNGAM().fit(X)
   print(model.causal_order_)
   print(model.adjacency_matrix_)

The fitted model exposes ``causal_order_``, ``adjacency_matrix_``, ``score_``, and ``intercept_``. The adjacency entry ``(child, parent)`` is the estimated effect from parent to child.

The mathematics in one paragraph
---------------------------------

OTLiNGAM assumes a linear acyclic structural equation model

.. math::

   X_j = \sum_{k \in \operatorname{Pa}(j)} B_{jk} X_k + \varepsilon_j.

For a candidate order, each variable is regressed on its predecessors. A correct order leaves an independent structural noise in every residual. The objective sums the squared one-dimensional Wasserstein distances between standardized residual distributions and :math:`\mathcal{N}(0, 1)`:

.. math::

   G(\sigma) = \sum_{j=1}^{d} \mathcal{W}_2\left(\operatorname{std}(R_j(\sigma)), \mathcal{N}(0,1)\right)^2.

Empirically, the distance is computed by sorting residuals and comparing them with Gaussian quantiles. The method and its assumptions are described in the paper, `Contrast-Free ICA and Causal Inference via Wasserstein Distances to the Gaussian <https://arxiv.org/abs/2607.12832>`_.

Choosing an estimator
---------------------

``GreedyOTLiNGAM`` is the default for larger systems. ``ExhaustiveOTLiNGAM`` searches globally over subset states and is intended for smaller dimensions because its cost grows exponentially. ``OTICALiNGAM`` uses OTICA with FastICA initialization before the standard ICA-LiNGAM identification steps.

For centered data, pass ``fit_intercept=False``. The default centers observations and stores the fitted intercepts.
