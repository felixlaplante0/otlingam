OT-LiNGAM
=========

**otlingam** is a Python package for causal discovery in linear non-Gaussian
structural equation models. It learns causal orders by maximizing the Wasserstein
non-Gaussianity of standardized regression residuals and estimates edge weights with
adaptive Lasso.

.. code-block:: bash

   pip install otlingam

See :doc:`quickstart` for a first example, or the :doc:`tutorial`. The package is
available on `PyPI <https://pypi.org/project/otlingam/>`_, and the method is described
in the `paper <https://arxiv.org/abs/2607.12832>`_.

Why OT-LiNGAM?
--------------

In a linear structural equation model, a correct causal order makes each variable's regression residual recover one independent structural noise. Non-Gaussianity makes those residuals distinguishable from residuals formed by mixing several noises. OT-LiNGAM turns that idea into an empirical objective based on sorted residuals and Gaussian quantiles.

The exhaustive estimator searches all subsets and gives a global order optimum for small systems. The greedy estimator trades that guarantee for a quadratic-time procedure. The ICA estimator adds the same Wasserstein-based source estimation to the familiar ICA-LiNGAM workflow. All three follow the scikit-learn conventions and expose causal orders, weighted adjacency matrices and intercepts; the exhaustive and greedy estimators also report their Wasserstein score.

Quick example
-------------

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

The :doc:`quickstart` explains the objective and how to choose an estimator, and the
:doc:`tutorial` compares them on a full example.

Learn
-----

.. grid:: 1 1 1 3
   :gutter: 3

   .. grid-item-card:: Quick start
      :link: quickstart.html

      Install the package, fit an estimator, and understand the mathematical objective.

   .. grid-item-card:: Tutorial notebook
      :link: tutorial.html

      Follow a complete synthetic example with plots and a comparison of the estimators.

Citation
--------

If you use OT-LiNGAM, please cite:

.. code-block:: bibtex

   @article{laplante2026otlingam,
     title   = {Contrast-Free ICA and Causal Inference via Wasserstein Distances
                to the Gaussian},
     author  = {Laplante, F{\'e}lix and Ambroise, Christophe and Humbert, Pierre},
     journal = {arXiv preprint arXiv:2607.12832},
     year    = {2026},
     doi     = {10.48550/arXiv.2607.12832}
   }

.. toctree::
   :hidden:

   quickstart
   highway
   tutorial
   modules
