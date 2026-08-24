from numbers import Integral
from typing import ClassVar, Self

import numpy as np
from otica import OTICA
from scipy.optimize import linear_sum_assignment  # type: ignore
from sklearn.utils._param_validation import Interval  # type: ignore
from sklearn.utils.validation import validate_data  # type: ignore

from ._base import BaseLiNGAM


def _search_causal_order(matrix: np.ndarray) -> list[np.integer] | None:
    """Finds a causal order in a matrix with an exact triangular structure.

    Args:
        matrix (np.ndarray): Square coefficient matrix whose zero rows identify source
            variables.

    Returns:
        list[np.integer] | None: Causal order from source to sink, or ``None`` when no
        complete order exists.
    """
    causal_order = []
    variable_count = matrix.shape[0]
    original_indices = np.arange(matrix.shape[0])
    while len(matrix) > 0:
        zero_rows = np.flatnonzero(np.sum(np.abs(matrix), axis=1) == 0.0)
        if len(zero_rows) == 0:
            break
        target = zero_rows[0]
        causal_order.append(original_indices[target])
        keep = np.delete(np.arange(len(matrix)), target)
        matrix = matrix[keep][:, keep]
        original_indices = original_indices[keep]

    if len(causal_order) != variable_count:
        return None

    return causal_order


def _estimate_causal_order(matrix: np.ndarray) -> list[np.integer] | None:
    """Estimates a causal order by progressively pruning weak coefficients.

    The function modifies ``matrix`` in place while searching for an approximately
    lower-triangular representation.

    Args:
        matrix (np.ndarray): Square coefficient matrix estimated by ICA.

    Returns:
        list[np.integer] | None: Causal order from source to sink, or ``None`` when no
        complete order can be recovered.
    """
    positions = np.argsort(np.abs(matrix), axis=None)
    positions = np.column_stack(np.unravel_index(positions, matrix.shape))
    initial_zeros = matrix.shape[0] * (matrix.shape[0] + 1) // 2
    for row, column in positions[:initial_zeros]:
        matrix[row, column] = 0.0

    causal_order = None
    for row, column in positions[initial_zeros:]:
        causal_order = _search_causal_order(matrix)
        if causal_order is not None:
            break
        matrix[row, column] = 0.0

    return causal_order


class OTICALiNGAM(BaseLiNGAM):
    """ICA-based LiNGAM using optimal transport ICA.

    This estimator learns a directed acyclic graph by estimating an unmixing matrix with
    ``OTICA``. The resulting matrix is permuted and scaled before a causal order and
    adjacency matrix are estimated using the standard ICA-LiNGAM identification steps.

    Optimization settings:
        - ``random_state``: Seed used by OTICA's random number generator.
        - ``max_iter``: Maximum number of OTICA optimization iterations.

    Attributes:
        random_state (int | None): Seed used by OTICA's random number generator.
        max_iter (int): Maximum number of OTICA optimization iterations.
        causal_order_ (list[np.integer]): Learned causal order from source to sink.
        adjacency_matrix_ (np.ndarray): Learned weighted adjacency matrix.
        intercept_ (np.ndarray): Intercepts of the structural equations.

    Examples:
        >>> from otlingam import OTICALiNGAM
        >>> model = OTICALiNGAM(random_state=0, max_iter=1000)
        >>> model.fit(X)
        >>> model.causal_order_
    """

    intercept_: np.ndarray

    _parameter_constraints: ClassVar[dict] = {
        "random_state": ["random_state"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, random_state: int | None = None, max_iter: int = 1000):
        """Initializes OTICALiNGAM.

        Args:
            random_state (int | None, optional): Seed used by OTICA. Defaults to None.
            max_iter (int, optional): Maximum OTICA iterations. Defaults to 1000.
        """
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X: np.typing.ArrayLike, y: None = None) -> Self:  # noqa: ARG002
        """Fits the model to the observations.

        Args:
            X (np.typing.ArrayLike): Training observations.
            y (None, optional): Ignored. Defaults to None.

        Returns:
            Self: The fitted estimator.
        """
        self._validate_params()
        X = np.asarray(validate_data(self, X, dtype=np.float64))  # type: ignore

        ica = OTICA(
            max_iter=self.max_iter,
            random_state=self.random_state,
        ).fit(X)  # type: ignore
        W_ica = ica.components_

        abs_W_ica = np.abs(W_ica)
        cost = np.full(abs_W_ica.shape, np.finfo(abs_W_ica.dtype).max)
        np.divide(1.0, abs_W_ica, out=cost, where=abs_W_ica > 0.0)
        _, col_index = linear_sum_assignment(cost)
        PW_ica = np.zeros_like(W_ica)
        PW_ica[col_index] = W_ica

        D = np.diag(PW_ica)[:, np.newaxis]
        if np.any(D == 0.0):
            raise ValueError("OTICA produced a singular unmixing permutation.")

        W_estimate = PW_ica / D
        B_estimate = np.eye(len(W_estimate)) - W_estimate

        causal_order = _estimate_causal_order(B_estimate)
        if causal_order is None:
            raise ValueError(
                "A causal order could not be estimated from the ICA result."
            )
        self.causal_order_ = causal_order
        self._estimate_adjacency_matrix(X)
        self.intercept_ = ica.mean_ - self.adjacency_matrix_ @ ica.mean_

        return self
