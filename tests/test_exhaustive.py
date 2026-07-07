"""Tests for exhaustive dynamic-programming internals."""

import numpy as np

from otlingam._exhaustive import (
    _causal_order,
    _cholesky_solve_norm_inplace,
    _compute_residuals,
    _get_A_num,
    _score,
    _sink_dp,
    _solve_coef,
)
from otlingam._utils import gauss_quantiles

from ._utils import linear_dag


def test_exhaustive_helpers_match_linear_regression_quantities():
    """Exercises exhaustive dynamic-programming helpers through Python tracing."""
    X, _ = linear_dag()
    X = X - X.mean(axis=0)
    cov_matrix = X.T @ X
    mask = 0b011
    target = 2
    d = X.shape[1]
    parent_count = 2

    A, k = _get_A_num.py_func(cov_matrix, mask, target, d)
    assert k == parent_count

    gram = cov_matrix[:2, :2]
    cross = cov_matrix[target, :2]
    expected_coef = np.linalg.solve(gram, cross)
    expected_residuals = X[:, target] - X[:, :2] @ expected_coef
    expected_rss = expected_residuals @ expected_residuals

    rss = _cholesky_solve_norm_inplace.py_func(A.copy(), k)
    factor = A.copy()
    _cholesky_solve_norm_inplace.py_func(factor, k)
    coef = _solve_coef.py_func(factor, k)
    residuals = _compute_residuals.py_func(X, target, mask, coef, d)

    assert np.isclose(rss, expected_rss)
    assert np.allclose(coef, expected_coef)
    assert np.allclose(residuals, expected_residuals)


def test_exhaustive_score_and_sink_dp_return_finite_order():
    """Exercises score and subset dynamic programming helpers."""
    X, _ = linear_dag()
    X = X - X.mean(axis=0)
    cov_matrix = X.T @ X
    quantiles = gauss_quantiles(X.shape[0])
    d = X.shape[1]

    score = _score.py_func(X, cov_matrix, quantiles, target=2, mask=0b011, d=d)
    sinks, total_score = _sink_dp.py_func(X, cov_matrix, quantiles, d)
    order = _causal_order(sinks, d)

    assert np.isfinite(score)
    assert np.isfinite(total_score)
    assert sorted(order.tolist()) == [0, 1, 2]
