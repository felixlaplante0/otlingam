import numpy as np
import pytest

from otlingam import ExhaustiveOTLiNGAM
from otlingam._exhaustive import (
    _MAX_DP_VARIABLES,
    _build_augmented_gram,
    _causal_order,
    _cholesky_solve_norm_inplace,
    _compute_residuals,
    _score,
    _sink_dp,
    _solve_coef,
)
from otlingam._utils import gauss_quantiles

from ._utils import linear_dag


def test_regression_helpers():
    """Exercises exhaustive dynamic-programming helpers through Python tracing."""
    X, _ = linear_dag()
    X = X - X.mean(axis=0)
    cov_matrix = X.T @ X
    mask = 0b11
    target = 2
    d = X.shape[1]
    parent_count = 2

    A, k = _build_augmented_gram.py_func(cov_matrix, mask, target, d)
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


def test_sink_dp():
    """Exercises score and subset dynamic programming helpers."""
    X, _ = linear_dag()
    X = X - X.mean(axis=0)
    cov_matrix = X.T @ X
    quantiles = gauss_quantiles(X.shape[0])
    d = X.shape[1]

    score = _score.py_func(X, cov_matrix, quantiles, target=2, mask=0b11, d=d)
    sinks, total_score = _sink_dp.py_func(X, cov_matrix, quantiles, d)
    order = _causal_order(sinks, d)

    assert np.isfinite(score)
    assert np.isfinite(total_score)
    assert sorted(order.tolist()) == [0, 1, 2]


def test_singular_residuals():
    """Checks that singular residual systems produce a non-finite score."""
    X = np.ones((4, 2))

    estimator = ExhaustiveOTLiNGAM().fit(X)

    assert not np.isfinite(estimator.score_)


def test_variable_limit():
    """Checks that oversized exhaustive problems fail before DP allocation."""
    X = np.ones((2, _MAX_DP_VARIABLES + 1))

    with pytest.raises(ValueError, match="at most"):
        ExhaustiveOTLiNGAM().fit(X)
