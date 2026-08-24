OTLiNGAM
========

.. raw:: html

   <section class="hero">
     <img class="hero-logo" src="_static/otlingam-logo.svg" alt="OTLiNGAM logo">
     <p class="eyebrow">CAUSAL DISCOVERY · OPTIMAL TRANSPORT</p>
     <h1>Find the order hidden in non-Gaussian data.</h1>
     <p class="hero-copy">OTLiNGAM learns causal structure in linear non-Gaussian models with exact one-dimensional Wasserstein objectives.</p>
     <div class="hero-actions">
       <a class="primary" href="quickstart.html">Get started</a>
       <a href="https://github.com/felixlaplante0/otlingam">View on GitHub</a>
     </div>
   </section>

.. raw:: html

   <div class="pypi-card">
     <div><span class="pypi-kicker">OPEN SOURCE PYTHON PACKAGE</span><strong>Install OTLiNGAM in seconds</strong><p>Works with NumPy, scikit-learn, and the LiNGAM ecosystem.</p></div>
     <a href="quickstart.html">Read the quick start</a>
   </div>

Highlights
----------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Wasserstein scoring
      :class-card: feature-card

      Compare standardized residuals with a Gaussian reference through the exact empirical one-dimensional :math:`W_2` distance.

   .. grid-item-card:: Three estimators
      :class-card: feature-card

      Choose exhaustive dynamic programming, scalable greedy ordering, or OTICA inside the classical ICA-LiNGAM pipeline.

   .. grid-item-card:: Familiar API
      :class-card: feature-card

      Estimators follow the scikit-learn conventions and expose causal orders, adjacency matrices, scores, and intercepts.

   .. grid-item-card:: Inspectable results
      :class-card: feature-card

      Inspect causal orders, weighted adjacency matrices, scores, and intercepts with familiar NumPy tools.

Why OTLiNGAM?
-------------

In a linear structural equation model, a correct causal order makes each variable's regression residual recover one independent structural noise. Non-Gaussianity makes those residuals distinguishable from residuals formed by mixing several noises. OTLiNGAM turns that idea into an empirical objective based on sorted residuals and Gaussian quantiles.

The exhaustive estimator searches all subsets and gives a global order optimum for small systems. The greedy estimator trades that guarantee for a quadratic-time procedure. The ICA estimator adds the same Wasserstein-based source estimation to the familiar ICA-LiNGAM workflow.

Learn
-----

.. grid:: 1 1 1 3
   :gutter: 3

   .. grid-item-card:: Quick start
      :link: quickstart
      :class-card: feature-card

      Install the package, fit an estimator, and understand the mathematical objective.

   .. grid-item-card:: Tutorial notebook
      :link: tutorial
      :class-card: feature-card

      Follow a complete synthetic example with plots and a comparison of the estimators.

.. raw:: html

   <p><a class="tutorial-link" href="https://github.com/felixlaplante0/otlingam/blob/main/examples/tutorial.ipynb">Open the tutorial notebook source on GitHub</a></p>

API reference
-------------

.. toctree::
   :maxdepth: 2
   :hidden:

   quickstart
   highway
   tutorial
   modules
