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
    """Estimates pruned regression coefficients with adaptive Lasso.

    Args:
        X (np.ndarray): Observation matrix with shape ``(n_samples, n_features)``.
        predictors (list[int]): Indices of candidate parent variables.
        target (int): Index of the response variable.

    Returns:
        np.ndarray: Regression coefficients corresponding to ``predictors``.
    """
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


class BaseLiNGAM(BaseEstimator):
    """Provides shared fitted state and edge estimation for LiNGAM estimators.

    Attributes:
        causal_order_ (list[int]): Estimated causal ordering from source to sink.
        adjacency_matrix_ (np.ndarray): Estimated weighted adjacency matrix.
    """

    def _estimate_adjacency_matrix(self, X: np.ndarray) -> Self:
        """Estimates graph edge weights for the current causal order.

        Args:
            X (np.ndarray): Observation matrix with shape ``(n_samples, n_features)``.

        Returns:
            Self: The fitted estimator with its adjacency matrix set.
        """
        self.adjacency_matrix_ = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
        for position in range(1, len(self.causal_order_)):
            target = self.causal_order_[position]
            predictors = self.causal_order_[:position]
            self.adjacency_matrix_[target, predictors] = _predict_adaptive_lasso(
                X,
                predictors,
                target,
            )

        return self
