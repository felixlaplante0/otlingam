PIT-CP
======

**pitcp** is a Python package for conformal prediction using **probability integral transform (PIT) pivotal scores**. Given any black-box nonconformity score, it fits a conditional density estimator on the score distribution and maps raw scores to PIT values, yielding valid marginal coverage at any user-specified level.

Features
--------

- **PIT Conformal Prediction**: Maps base nonconformity scores through a learned conditional CDF, producing asymptotically exact conditional coverage.
- **Model-agnostic**: Works with any callable nonconformity score `s(x, y)`, including distance, residual, or likelihood-based scores.
- **Flexible Density Estimation**: Supports normalizing flows and mixture density networks from the `zuko <https://github.com/probabilists/zuko>`_ library.
- **Marginal Coverage Guarantee**: Provably valid conformal coverage at any target level via finite-sample calibration.
- **scikit-learn**: Native `BaseEstimator` integration with a familiar `fit` / `conformalize` / `predict` API.

Installation
------------

You can install the package via pip:

.. code-block:: bash

   python -m pip install pitcp

Usage
-----

Example:

.. code-block:: python

    import torch
    import zuko
    from pitcp import PITCP


    def std(x):
        return torch.where((x > -0.9) & (x < 0.9), torch.cos(torch.pi * x / 2), 1.0)


    def gen_data(n):
        x = torch.rand(n, 1) * 2 - 1
        return x, torch.randn(n, 1) * std(x)


    torch.manual_seed(42)

    (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = [
        gen_data(5000) for _ in range(3)
    ]


    # Define a nonconformity score
    def score(x, y):
        return y.abs()


    # Build a normalizing flow density estimator
    model = zuko.flows.NSF(features=1, context=1, bins=4, hidden_features=(32, 32, 32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Compute nonconformity scores
    s_train = score(X_train, y_train)
    s_cal = score(X_cal, y_cal)
    s_test = score(X_test, y_test)

    # Fit and conformalize
    pitcp = PITCP(model, optimizer, n_epochs=10, batch_size=128)
    pitcp.fit(X_train, s_train)
    pitcp.conformalize(X_cal, s_cal)

    # Predict conformal regions (max score thresholds) at multiple quantiles
    limits = pitcp.predict(X_test, quantile=[0.7, 0.8, 0.9])

    # Predict conformal coverage at multiple quantiles (single float accepted)
    covered = pitcp.predict_coverage(X_test, s_test, quantile=[0.7, 0.8, 0.9])
    print(f"Empirical coverages: {covered.mean(axis=0)}")

API Reference
-------------

.. autoclass:: pitcp.PITCP
   :members:
   :undoc-members:
   :show-inheritance:
