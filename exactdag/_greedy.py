"""Greedy quadratic-time causal ordering via the squared W2 distance score."""

from __future__ import annotations

from typing import Self, cast

import numpy as np
from numba import njit  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils._param_validation import validate_params  # type: ignore
from sklearn.utils.validation import validate_data  # type: ignore

from ._base import _gauss_quantiles, _recover_weights, _score


@njit(cache=True, fastmath=True)  # type: ignore
def _greedy_order(
    X: np.ndarray, cov_matrix: np.ndarray, q: np.ndarray, d: int
) -> tuple[np.ndarray, float]:
    """Builds a causal order greedily by most non-Gaussian residual.

    At every step, among the not-yet-placed variables, each candidate is regressed on
    the already-placed predecessors and its standardized residual is scored by the
    squared W2 distance to ``N(0, 1)`` (the same :func:`_score` used by the exact
    dynamic program). The candidate with the largest score is appended next.

    Args:
        X (np.ndarray): Input data.
        cov_matrix (np.ndarray): Covariance matrix (``X.T @ X``).
        q (np.ndarray): Precomputed N(0, 1) quantiles.
        d (int): Number of variables.

    Returns:
        tuple[np.ndarray, float]: Causal order from source to sink and the total
            score, that is the sum of the chosen residual scores.
    """
    order = np.empty(d, dtype=np.int32)
    placed_mask = 0
    total = 0.0

    for t in range(d):
        best_score = -np.inf
        best_j = -1
        for j in range(d):
            if (placed_mask >> j) & 1:
                continue
            s = _score(X, cov_matrix, q, j, placed_mask, d)
            if s > best_score:
                best_score = s
                best_j = j
        order[t] = best_j
        placed_mask |= 1 << best_j
        total += best_score

    return order, total


class GreedyDAGNegW2(BaseEstimator):
    """Greedy quadratic-time causal discovery via the squared W2 distance score.

    A greedy, polynomial-time alternative to the exact subset dynamic programming of
    :class:`~exactdag.ExactDAG`. It builds the causal order from source to sink one
    variable at a time. At every step, among the not-yet-placed variables, it regresses
    each on the placed predecessors and selects the one whose standardized residual is
    the most distant from the standard Gaussian in squared Wasserstein distance, that is
    the most "exogenous given the prefix" variable.

    Both estimators maximize the same order objective, the sum of the residual scores,
    but :class:`GreedyDAGNegW2` makes ``d`` locally optimal decisions instead of the
    exact dynamic program over ``2**d`` subsets. It runs in ``O(d**2)`` score
    evaluations and is therefore practical in larger dimension. When the structural
    noises are not too heterogeneous in their Wasserstein non-Gaussianity, the greedy
    rule recovers a topological order; under strongly heterogeneous noises a deep but
    highly non-Gaussian noise can mislead an early choice, where the exact estimator is
    more robust. Once the order is recovered, edge weights are estimated by adaptive
    lasso, exactly as in :class:`~exactdag.ExactDAG`.

    Attributes:
        fit_intercept (bool): Whether to center the data before fitting.
        causal_order_ (np.ndarray): Learned causal order from source to sink.
        adjacency_matrix_ (np.ndarray): Learned weighted adjacency matrix.
        intercept_ (np.ndarray): Intercepts of the regression models. Available only
            when ``fit_intercept`` is ``True``.
        score_ (float): Total squared Wasserstein distance-based score of the order.

    Examples:
        >>> from exactdag import GreedyDAGNegW2
        >>> model = GreedyDAGNegW2(fit_intercept=True)
        >>> model.fit(X)
        >>> model.causal_order_
    """

    fit_intercept: bool
    causal_order_: np.ndarray
    adjacency_matrix_: np.ndarray
    intercept_: np.ndarray
    score_: float

    @validate_params({"fit_intercept": [bool]}, prefer_skip_nested_validation=True)
    def __init__(self, fit_intercept: bool = True) -> None:
        """Initialize GreedyDAGNegW2.

        Args:
            fit_intercept (bool, optional): Whether to center the data. Defaults to
                True.
        """
        self.fit_intercept = fit_intercept

    @validate_params(
        {"X": ["array-like"], "y": [None]},
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: np.typing.ArrayLike, y: None = None) -> Self:  # noqa: ARG002
        """Fits the GreedyDAGNegW2 algorithm.

        Args:
            X (np.typing.ArrayLike): Input data.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            GreedyDAGNegW2: The fitted estimator.
        """
        X = np.asarray(validate_data(self, X, dtype=np.float64))  # type: ignore
        n, d = X.shape

        if self.fit_intercept:
            shift = X.mean(axis=0)
            X = X - shift  # type: ignore

        cov_matrix = cast(np.ndarray, X.T @ X)  # type: ignore

        q = _gauss_quantiles(n)  # type: ignore

        self.causal_order_, self.score_ = _greedy_order(X, cov_matrix, q, d)  # type: ignore
        self.adjacency_matrix_ = _recover_weights(self.causal_order_, X, d)  # type: ignore

        if self.fit_intercept:
            self.intercept_ = shift - self.adjacency_matrix_ @ shift  # type: ignore

        return self
