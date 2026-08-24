"""Shared estimator behavior for OTLiNGAM models."""

from typing import Self

import numpy as np
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.linear_model import LassoLarsIC, LinearRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


def _predict_adaptive_lasso(
    X: np.ndarray,
    predictors: list[int],
    target: int,
) -> np.ndarray:
    """Estimates pruned regression coefficients with adaptive Lasso."""
    X_std = StandardScaler().fit_transform(X)
    linear_model = LinearRegression().fit(X_std[:, predictors], X_std[:, target])
    weights = np.abs(linear_model.coef_)
    lasso_model = LassoLarsIC(criterion="bic").fit(
        X_std[:, predictors] * weights,
        X_std[:, target],
    )
    selected = np.abs(lasso_model.coef_ * weights) > 0.0
    coefficients = np.zeros_like(lasso_model.coef_)
    if np.any(selected):
        predictor_indices = np.asarray(predictors)[selected]
        coefficients[selected] = (
            LinearRegression()
            .fit(
                X[:, predictor_indices],
                X[:, target],
            )
            .coef_
        )

    return coefficients


class _BaseOTLiNGAM(BaseEstimator):
    """Provides fitted graph state and adaptive-Lasso edge estimation."""

    @property
    def causal_order_(self) -> list[int] | None:
        """Returns the estimated causal order."""
        return getattr(self, "_causal_order", None)

    @property
    def adjacency_matrix_(self) -> np.ndarray | None:
        """Returns the estimated weighted adjacency matrix."""
        return getattr(self, "_adjacency_matrix", None)

    def _estimate_adjacency_matrix(self, X: np.ndarray) -> Self:
        adjacency_matrix = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
        for position in range(1, len(self._causal_order)):
            target = self._causal_order[position]
            predictors = self._causal_order[:position]
            adjacency_matrix[target, predictors] = _predict_adaptive_lasso(
                X,
                predictors,
                target,
            )
        self._adjacency_matrix = adjacency_matrix

        return self
