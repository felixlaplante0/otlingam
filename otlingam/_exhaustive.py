from typing import Self, cast

import numpy as np
from lingam.base import _BaseLiNGAM
from numba import njit  # type: ignore
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import validate_params  # type: ignore
from sklearn.utils.validation import validate_data

from ._utils import gauss_quantiles, recover_weights


@njit(cache=True, inline="always")  # type: ignore
def _get_A_num(
    cov_matrix: np.ndarray, mask: int, target: int, d: int
) -> tuple[np.ndarray, int]:
    """Gets the packed lower-triangular system matrix A and number of parents.

    Args:
        cov_matrix (np.ndarray): Covariance matrix.
        mask (int): Bitmask encoding the predictor variables.
        target (int): Index of the response variable.
        d (int): Number of variables.

    Returns:
        tuple[np.ndarray, int]: Packed lower-triangular matrix and number of parents.
    """
    parents = np.empty(d - 1, dtype=np.int32)
    A = np.empty(d * (d + 1) // 2, dtype=np.float64)
    i = k = 0

    while mask:
        if mask & 1:
            kk = k * (k + 1) // 2
            for j in range(k):
                A[kk + j] = cov_matrix[i, parents[j]]
            A[kk + k] = cov_matrix[i, i]
            parents[k] = i
            k += 1
        mask >>= 1
        i += 1

    kk = k * (k + 1) // 2
    for j in range(k):
        A[kk + j] = cov_matrix[target, parents[j]]
    A[kk + k] = cov_matrix[target, target]

    return A, k


@njit(cache=True, inline="always")  # type: ignore
def _cholesky_solve_norm_inplace(A: np.ndarray, k: int) -> float:
    """Computes the residual sum of squares using an in-place Cholesky decomposition.

    Args:
        A (np.ndarray): Packed lower-triangular Gram matrix.
        k (int): Number of regression coefficients.

    Returns:
        float: Residual sum of squares.
    """
    for i in range(k + 1):
        ii = i * (i + 1) // 2
        for j in range(i):
            ij = ii + j
            for l in range(j):  # noqa: E741
                A[ij] -= A[ii + l] * A[j * (j + 1) // 2 + l]
            A[ij] /= A[j * (j + 1) // 2 + j]
        j = i
        ij = ii + j
        for l in range(j):  # noqa: E741
            A[ij] -= A[ii + l] * A[j * (j + 1) // 2 + l]
        if A[ij] <= 0.0:
            raise ValueError("X must not induce a singular residual system.")
        A[ij] = np.sqrt(A[ij])

    return A[k * (k + 3) // 2] ** 2


@njit(cache=True, inline="always")  # type: ignore
def _solve_coef(A: np.ndarray, k: int) -> np.ndarray:
    """Solves for regression coefficients using the Cholesky factor.

    Args:
        A (np.ndarray): Packed Cholesky factor and transformed response vector.
        k (int): Number of regression coefficients.

    Returns:
        np.ndarray: Regression coefficients.
    """
    coef = np.empty(k, dtype=np.float64)
    kk = k * (k + 1) // 2
    for i in range(k):
        coef[i] = A[kk + i]

    for i in range(k - 1, -1, -1):
        for j in range(i + 1, k):
            coef[i] -= A[j * (j + 1) // 2 + i] * coef[j]
        coef[i] /= A[i * (i + 1) // 2 + i]
    return coef


@njit(cache=True, inline="always")  # type: ignore
def _compute_residuals(
    X: np.ndarray, target: int, mask: int, coef: np.ndarray, d: int
) -> np.ndarray:
    """Computes residuals from regression coefficients.

    Args:
        X (np.ndarray): Input data.
        target (int): Index of the response variable.
        mask (int): Bitmask encoding the predictor variables.
        coef (np.ndarray): Regression coefficients.
        d (int): Number of variables.

    Returns:
        np.ndarray: Regression residuals.
    """
    z = X[:, target].copy()
    idx = 0
    for i in range(d):
        if (mask >> i) & 1:
            z -= coef[idx] * X[:, i]
            idx += 1
    return z


@njit(cache=True, inline="always")  # type: ignore
def _score(
    X: np.ndarray,
    cov_matrix: np.ndarray,
    quantiles: np.ndarray,
    target: int,
    mask: int,
    d: int,
) -> float:
    """Calculates the squared W2 distance-based score.

    Args:
        X (np.ndarray): Input data.
        cov_matrix (np.ndarray): Covariance matrix.
        quantiles (np.ndarray): Precomputed N(0, 1) quantiles.
        target (int): Index of the response variable.
        mask (int): Bitmask encoding the predictor variables.
        d (int): Number of variables.

    Returns:
        float: Squared W2 distance-based score.
    """
    n = X.shape[0]

    A, k = _get_A_num(cov_matrix, mask, target, d)
    rss = _cholesky_solve_norm_inplace(A, k)
    coef = _solve_coef(A, k)

    z = _compute_residuals(X, target, mask, coef, d)
    z /= np.sqrt(rss / n)
    z.sort()

    return np.mean((z - quantiles) ** 2)  # type: ignore


@njit(cache=True, fastmath=True)  # type: ignore
def _sink_dp(
    X: np.ndarray, cov_matrix: np.ndarray, quantiles: np.ndarray, d: int
) -> tuple[np.ndarray, float]:
    """Finds the optimal sink node for every subset via dynamic programming.

    Args:
        X (np.ndarray): Input data.
        cov_matrix (np.ndarray): Covariance matrix.
        quantiles (np.ndarray): Precomputed N(0, 1) quantiles.
        d (int): Number of variables.

    Returns:
        tuple[np.ndarray, float]: Optimal sink index for each subset and total score.
    """
    n = 1 << d
    H = np.zeros(n, dtype=np.float64)
    sinks = np.full(n, -1, dtype=np.int32)

    for mask in range(1, n):
        cur_best_score = -np.inf
        cur_best_sink = -1
        bits = mask
        s = 0
        while bits:
            if bits & 1:
                prev_mask = mask ^ (1 << s)
                score = H[prev_mask] + _score(X, cov_matrix, quantiles, s, prev_mask, d)
                if score > cur_best_score:
                    cur_best_score = score
                    cur_best_sink = s
            bits >>= 1
            s += 1
        H[mask] = cur_best_score
        sinks[mask] = cur_best_sink

    return sinks, H[n - 1]


def _causal_order(sinks: np.ndarray, d: int) -> np.ndarray:
    """Recovers the causal order from source to sink.

    Args:
        sinks (np.ndarray): Optimal sink index for each subset.
        d (int): Number of variables.

    Returns:
        np.ndarray: Causal order from source to sink.
    """
    order = np.empty(d, dtype=int)
    mask = (1 << d) - 1

    for i in range(d):
        s = sinks[mask]
        order[i] = s
        mask ^= 1 << s

    return order[::-1]


class ExhaustiveOTLiNGAM(_BaseLiNGAM, BaseEstimator):
    """Exhaustive score-based causal discovery via subset dynamic programming.

    This estimator learns a directed acyclic graph by finding the causal ordering that
    maximizes a squared Wasserstein distance-based score. For each candidate sink, all
    preceding variables in the ordering are used as its parent set.

    The optimal ordering is found exhaustively using subset dynamic programming.
    Regression residuals are standardized and compared with standard normal quantiles
    to compute the score. Once the ordering is recovered, edge weights are estimated
    using adaptive lasso regression.

    Data preprocessing settings:
        - `fit_intercept`: Whether to center the data before fitting. Centering also
          enables estimation of an intercept for each variable.

    Attributes:
        fit_intercept (bool): Whether to center the data before fitting.
        _causal_order (list[np.integer] | None): Internal causal ordering. None before
            fitting.
        _adjacency_matrix (np.ndarray | None): Internal weighted adjacency matrix.
            None before fitting.
        causal_order_ (list[np.integer]): Learned causal order from source to sink.
        adjacency_matrix_ (np.ndarray): Learned weighted adjacency matrix.
        intercept_ (np.ndarray): Intercepts of the regression models. Available only
            when `fit_intercept` is `True`.
        score_ (float): Squared Wasserstein distance-based score of the learned DAG.

    Examples:
        >>> from otlingam import ExhaustiveOTLiNGAM
        >>> model = ExhaustiveOTLiNGAM(fit_intercept=True)
        >>> model.fit(X)
        >>> model.causal_order_
    """

    fit_intercept: bool
    intercept_: np.ndarray
    score_: float

    @validate_params({"fit_intercept": [bool]}, prefer_skip_nested_validation=True)
    def __init__(self, fit_intercept: bool = True) -> None:
        """Initializes ExhaustiveOTLiNGAM.

        Args:
            fit_intercept (bool, optional): Whether to center the data. Defaults to
                True.
        """
        super().__init__()
        self.fit_intercept = fit_intercept

    @validate_params(
        {"X": ["array-like"], "y": [None]},
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: np.typing.ArrayLike, y: None = None) -> Self:  # noqa: ARG002
        """Fits the ExhaustiveOTLiNGAM algorithm.

        Args:
            X (np.typing.ArrayLike): Input data.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            ExhaustiveOTLiNGAM: The fitted estimator.
        """
        X = cast(
            np.ndarray,
            validate_data(self, X, dtype=np.float64),  # type: ignore
        )
        n, d = X.shape

        if self.fit_intercept:
            shift = X.mean(axis=0)
            X = X - shift  # type: ignore

        cov_matrix = cast(np.ndarray, X.T @ X)  # type: ignore

        quantiles = gauss_quantiles(n)  # type: ignore

        sinks, self.score_ = _sink_dp(X, cov_matrix, quantiles, d)  # type: ignore
        order = _causal_order(sinks, d)
        self._causal_order = list(order)
        self._adjacency_matrix = recover_weights(order, X, d)  # type: ignore

        if self.fit_intercept:
            self.intercept_ = shift - self._adjacency_matrix @ shift  # type: ignore
        else:
            self.__dict__.pop("intercept_", None)

        return self
