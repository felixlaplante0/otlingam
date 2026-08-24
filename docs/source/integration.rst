Integration
===========

OTLiNGAM estimators follow the scikit-learn estimator protocol. They can be cloned, configured with ``get_params`` and ``set_params``, and placed after preprocessing steps in a pipeline.

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from otlingam import GreedyOTLiNGAM

   pipeline = Pipeline(
       [
           ("scale", StandardScaler()),
           ("causal", GreedyOTLiNGAM(fit_intercept=False)),
       ]
   )
   pipeline.fit(X)
   order = pipeline.named_steps["causal"].causal_order_

The fitted estimator exposes the same core outputs used by LiNGAM-style tooling:

* ``causal_order_`` is an order from source to sink
* ``adjacency_matrix_`` stores edge weights with rows as children and columns as parents
* ``score_`` stores the Wasserstein order score when the estimator uses one
* ``intercept_`` stores fitted intercepts when intercept fitting is enabled

You can inspect the result with NumPy or pass the adjacency matrix to downstream graph visualisation code:

.. code-block:: python

   import numpy as np

   edges = np.argwhere(np.abs(model.adjacency_matrix_) > 1e-8)
   for child, parent in edges:
       weight = model.adjacency_matrix_[child, parent]
       print(f"{parent} -> {child}: {weight:.3f}")

The `tutorial notebook <https://github.com/felixlaplante0/otlingam/blob/main/examples/tutorial.ipynb>`_ shows a complete workflow with simulated data and matrix plots.
