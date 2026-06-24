"""Greedy score-based DAG learning by sequential source removal."""

from __future__ import annotations

from typing import Self

import numpy as np
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils._param_validation import validate_params  # type: ignore
from sklearn.utils.validation import validate_data  # type: ignore

from ._utils import gauss_quantiles, recover_weights


class GreedyW2(BaseEstimator):
    """Greedy score-based causal discovery by sequential source removal.

    This estimator repeatedly selects the most non-Gaussian standardized residual as
    the next source in the causal order. It then removes the source's linear effect
    from every remaining variable. Once the ordering is recovered, edge weights are
    estimated using adaptive lasso regression.

    Data preprocessing settings:
        - `fit_intercept`: Whether to center the data before fitting. Centering also
          enables estimation of an intercept for each variable.

    Attributes:
        fit_intercept (bool): Whether to center the data before fitting.
        causal_order_ (np.ndarray): Learned causal order from source to sink.
        adjacency_matrix_ (np.ndarray): Learned weighted adjacency matrix.
        intercept_ (np.ndarray): Intercepts of the regression models. Available only
            when `fit_intercept` is `True`.
        score_ (float): Sum of the selected squared Wasserstein scores.

    Examples:
        >>> from otdag import GreedyW2
        >>> model = GreedyW2(fit_intercept=True)
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
        """Initializes GreedyW2.

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
        """Fits the GreedyW2 algorithm.

        Args:
            X (np.typing.ArrayLike): Input data.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            GreedyW2: The fitted estimator.

        Raises:
            ValueError: If a residual has zero variance.
        """
        X = np.asarray(validate_data(self, X, dtype=np.float64))
        n, d = X.shape

        if self.fit_intercept:
            shift = X.mean(axis=0)
            X = X - shift

        residuals = X.copy()
        remaining = list(range(d))
        order = np.empty(d, dtype=int)
        quantiles = gauss_quantiles(n)
        score = 0.0

        for t in range(d):
            current = residuals[:, remaining]
            scales = np.sqrt(np.mean(current**2, axis=0))
            if np.any(scales == 0.0):
                raise ValueError("X must not contain a constant residual.")

            standardized = current / scales
            scores = np.mean((np.sort(standardized, axis=0) - quantiles[:, None]) ** 2, axis=0)
            source_index = np.argmax(scores)
            source = remaining.pop(source_index)
            order[t] = source
            score += scores[source_index]

            if not remaining:
                break

            source_residual = residuals[:, source]
            effects = source_residual @ residuals[:, remaining] / (
                source_residual @ source_residual
            )
            residuals[:, remaining] -= np.outer(source_residual, effects)

        self.causal_order_ = order
        self.adjacency_matrix_ = recover_weights(order, X, d)
        self.score_ = float(score)

        if self.fit_intercept:
            self.intercept_ = shift - self.adjacency_matrix_ @ shift

        return self
