from typing import ClassVar, Self, cast

import numpy as np
from sklearn.utils._param_validation import validate_params  # type: ignore
from sklearn.utils.validation import validate_data  # type: ignore

from ..utils._wasserstein import gauss_quantiles
from ._base import BaseLiNGAM
from ._exhaustive_kernel import _sink_dp

_MAX_DP_VARIABLES = 31


def _causal_order(sinks: np.ndarray, d: int) -> np.ndarray:
    """Reconstructs a causal order from dynamic-programming sink choices.

    Args:
        sinks (np.ndarray): Best sink index for each encoded variable subset.
        d (int): Number of variables in the graph.

    Returns:
        np.ndarray: Causal order from source to sink.
    """
    order = np.empty(d, dtype=int)
    mask = (1 << d) - 1
    for i in range(d):
        sink = sinks[mask]
        order[i] = sink
        mask ^= 1 << sink
    return order[::-1]


class ExhaustiveOTLiNGAM(BaseLiNGAM):
    """Exhaustive score-based causal discovery."""

    fit_intercept: bool
    intercept_: np.ndarray
    score_: float
    _parameter_constraints: ClassVar[dict] = {"fit_intercept": ["boolean"]}

    def __init__(self, fit_intercept: bool = True):
        super().__init__()
        self.fit_intercept = fit_intercept

    @validate_params(
        {"X": ["array-like"], "y": [None]},
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: np.typing.ArrayLike, y: None = None) -> Self:  # noqa: ARG002
        self._validate_params()
        X = np.asarray(validate_data(self, X, dtype=np.float64))  # type: ignore
        n, d = X.shape
        if d > _MAX_DP_VARIABLES:
            raise ValueError(
                "ExhaustiveOTLiNGAM supports at most "
                f"{_MAX_DP_VARIABLES} variables because its subset dynamic "
                "program stores 2 ** d states."
            )
        if self.fit_intercept:
            shift = X.mean(axis=0)
            X = X - shift  # type: ignore

        cov_matrix = cast(np.ndarray, X.T @ X)  # type: ignore
        quantiles = gauss_quantiles(n)  # type: ignore
        sinks, self.score_ = _sink_dp(X, cov_matrix, quantiles, d)
        self.causal_order_ = list(_causal_order(sinks, d))
        self._estimate_adjacency_matrix(X)

        if self.fit_intercept:
            self.intercept_ = shift - self.adjacency_matrix_ @ shift
        else:
            self.__dict__.pop("intercept_", None)
        return self
