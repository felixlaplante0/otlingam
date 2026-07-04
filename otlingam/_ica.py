from typing import Self, cast

import numpy as np
from lingam import OTICALiNGAM as BaseICALiNGAM
from otica import OTICA
from scipy.optimize import linear_sum_assignment  # type: ignore
from sklearn.utils.validation import validate_data  # type: ignore


class OTICALiNGAM(BaseICALiNGAM):
    """ICA-based LiNGAM using optimal transport ICA.

    This estimator learns a directed acyclic graph by estimating an unmixing matrix
    with `OTICA`. The resulting matrix is permuted and scaled before a causal order
    and adjacency matrix are estimated using ICA-LiNGAM's existing implementation.

    Optimization settings:
        - `random_state`: Seed used by OTICA's random number generator.
        - `max_iter`: Maximum number of OTICA optimization iterations.

    Attributes:
        _random_state (int | None): Seed used by OTICA's random number generator.
        _max_iter (int): Maximum number of OTICA optimization iterations.
        _causal_order (list[np.integer] | None): Internal causal ordering. None before
            fitting.
        _adjacency_matrix (np.ndarray | None): Internal weighted adjacency matrix.
            None before fitting.
        causal_order_ (list[np.integer]): Learned causal order from source to sink.
        adjacency_matrix_ (np.ndarray): Learned weighted adjacency matrix.

    Examples:
        >>> from otlingam import OTICALiNGAM
        >>> model = OTICALiNGAM(random_state=0, max_iter=1000)
        >>> model.fit(X)
        >>> model.causal_order_
    """

    def fit(self, X: np.typing.ArrayLike) -> Self:
        """Fits the model to the observations.

        Args:
            X (np.typing.ArrayLike): Training observations.

        Returns:
            Self: The fitted estimator.
        """
        X = cast(
            np.ndarray,
            validate_data(self, X, dtype=np.float64),  # type: ignore
        )

        ica = OTICA(
            max_iter=self._max_iter,
            random_state=self._random_state,  # type: ignore
        ).fit(X)  # type: ignore
        W_ica = ica.components_

        _, col_index = linear_sum_assignment(1 / np.abs(W_ica))
        PW_ica = np.zeros_like(W_ica)
        PW_ica[col_index] = W_ica

        D = np.diag(PW_ica)[:, np.newaxis]
        W_estimate = PW_ica / D
        B_estimate = np.eye(len(W_estimate)) - W_estimate

        self._causal_order = self._estimate_causal_order(B_estimate)

        return self._estimate_adjacency_matrix(X)