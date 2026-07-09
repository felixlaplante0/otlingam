import numpy as np
from lingam.utils import predict_adaptive_lasso  # type: ignore
from scipy.special import ndtri  # type: ignore
from sklearn.utils.validation import check_array, column_or_1d  # type: ignore


def gauss_quantiles(n: int) -> np.ndarray:
    """Computes standard-normal means over equal-probability quantile bins.

    Args:
        n (int): Number of quantile bins.

    Returns:
        np.ndarray: Mean standard-normal quantile in each bin.
    """
    z = ndtri(np.linspace(0.0, 1.0, n + 1))
    phi = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)  # type: ignore
    phi[[0, -1]] = 0.0
    return n * (phi[:-1] - phi[1:])


def recover_weights(order: np.ndarray, X: np.ndarray, d: int) -> np.ndarray:
    """Recovers regression edge weights given the causal order and parent sets.

    Args:
        order (np.ndarray): Causal order.
        X (np.ndarray): Data matrix.
        d (int): Number of variables.

    Returns:
        np.ndarray: Weight matrix.
    """
    W = np.zeros((d, d), dtype=np.float64)

    for t in range(1, d):
        j = order[t]
        parents = order[:t]
        W[j, parents] = predict_adaptive_lasso(X, parents, j)

    return W


def disorder(
    true_adjacency_matrix: np.typing.ArrayLike,
    pred_causal_order: np.typing.ArrayLike,
) -> int:
    r"""Counts true edges reversed by a causal order.

    Let :math:`\hat{\sigma}` be the estimated order. The disorder is given by

    .. math::

        \mathrm{dis}(\hat{\sigma}) = \#\left\{ (k, j) : B^\star_{jk} \neq 0 \text{ and }
        \hat{\sigma}^{-1}(k) > \hat{\sigma}^{-1}(j) \right\}

    It is zero exactly when `pred_causal_order` is a topological order of the true DAG.

    Args:
        true_adjacency_matrix (np.typing.ArrayLike): Ground-truth weighted adjacency
            matrix whose entry :math:`B_{jk}` represents the edge :math:`k \to j`.
        pred_causal_order (np.typing.ArrayLike): Estimated node permutation from source
            to sink.

    Returns:
        int: Number of reversed true edges.

    Raises:
        ValueError: If the matrix is not square or `pred_causal_order` is not a
            permutation.
    """
    true_edges = check_array(true_adjacency_matrix)
    pred_order = column_or_1d(pred_causal_order, dtype=int)  # type: ignore

    if true_edges.shape[0] != true_edges.shape[1]:
        raise ValueError(
            "true_adjacency_matrix must be a square array, "
            f"got shape {true_edges.shape}."
        )

    d = true_edges.shape[0]
    if not np.array_equal(np.sort(pred_order), np.arange(d)):
        raise ValueError("pred_causal_order must be a permutation of range(d).")

    pos = np.empty(d, dtype=np.int64)
    pos[pred_order] = np.arange(d)

    child, parent = np.nonzero(true_edges)
    return int(np.sum(pos[parent] > pos[child]))
