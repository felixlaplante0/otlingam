import numpy as np
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.utils.validation import check_array, column_or_1d  # type: ignore


def _check_adjacency_matrices(
    adjacency_matrix_true: np.typing.ArrayLike,
    adjacency_matrix_pred: np.typing.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Validates and binarizes a pair of adjacency matrices.

    Args:
        adjacency_matrix_true (np.typing.ArrayLike): Ground-truth weighted adjacency
            matrix.
        adjacency_matrix_pred (np.typing.ArrayLike): Estimated weighted adjacency
            matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: Binary true and estimated adjacency matrices.

    Raises:
        ValueError: If either matrix is not square or the shapes differ.
    """
    edges_true = check_array(adjacency_matrix_true) != 0.0
    edges_pred = check_array(adjacency_matrix_pred) != 0.0

    if edges_true.shape[0] != edges_true.shape[1]:
        raise ValueError(
            "adjacency_matrix_true must be a square array, "
            f"got shape {edges_true.shape}."
        )
    if edges_pred.shape[0] != edges_pred.shape[1]:
        raise ValueError(
            "adjacency_matrix_pred must be a square array, "
            f"got shape {edges_pred.shape}."
        )
    if edges_true.shape != edges_pred.shape:
        raise ValueError(
            "adjacency_matrix_true and adjacency_matrix_pred must have the same "
            f"shape, got {edges_true.shape} and {edges_pred.shape}."
        )

    return edges_true, edges_pred


def _off_diagonal_entries(adjacency_matrix: np.ndarray) -> np.ndarray:
    """Returns off-diagonal directed edge indicators as a flat array.

    Args:
        adjacency_matrix (np.ndarray): Binary square adjacency matrix.

    Returns:
        np.ndarray: Flattened off-diagonal entries.
    """
    return adjacency_matrix[~np.eye(adjacency_matrix.shape[0], dtype=bool)]


def _pair_states(adjacency_matrix: np.ndarray) -> np.ndarray:
    r"""Encodes unordered node pairs by their directed edge state.

    Args:
        adjacency_matrix (np.ndarray): Binary square adjacency matrix in the LiNGAM
            convention, where entry `(j, k)` represents the edge :math:`k \to j`.

    Returns:
        np.ndarray: Categorical edge states for unordered node pairs.
    """
    child, parent = np.tril_indices(adjacency_matrix.shape[0], k=-1)
    return adjacency_matrix[child, parent] + 2 * adjacency_matrix[parent, child]


def disorder(
    adjacency_matrix_true: np.typing.ArrayLike,
    causal_order_pred: np.typing.ArrayLike,
) -> int:
    r"""Counts true edges reversed by a causal order.

    Let :math:`\hat{\sigma}` be the estimated order. The disorder is given by

    .. math::

        \begin{aligned}
        \mathrm{dis}(\hat{\sigma}) &= \#\left\{ (k, j) : B^\star_{jk} \neq 0, \\
        &\quad \hat{\sigma}^{-1}(k) > \hat{\sigma}^{-1}(j) \right\}.
        \end{aligned}

    It is zero exactly when `causal_order_pred` is a topological order of the true DAG.

    Args:
        adjacency_matrix_true (np.typing.ArrayLike): Ground-truth weighted adjacency
            matrix whose entry :math:`B_{jk}` represents the edge :math:`k \to j`.
        causal_order_pred (np.typing.ArrayLike): Estimated node permutation from source
            to sink.

    Returns:
        int: Number of reversed true edges.

    Raises:
        ValueError: If the matrix is not square or `causal_order_pred` is not a
            permutation.
    """
    edges_true = check_array(adjacency_matrix_true)
    order_pred = column_or_1d(causal_order_pred, dtype=int)  # type: ignore

    if edges_true.shape[0] != edges_true.shape[1]:
        raise ValueError(
            "adjacency_matrix_true must be a square array, "
            f"got shape {edges_true.shape}."
        )

    d = edges_true.shape[0]
    if not np.array_equal(np.sort(order_pred), np.arange(d)):
        raise ValueError("causal_order_pred must be a permutation of range(d).")

    pos = np.empty(d, dtype=np.int64)
    pos[order_pred] = np.arange(d)

    child, parent = np.nonzero(edges_true)
    return int(np.sum(pos[parent] > pos[child]))


def shd(
    adjacency_matrix_true: np.typing.ArrayLike,
    adjacency_matrix_pred: np.typing.ArrayLike,
) -> int:
    r"""Computes structural Hamming distance between two directed graphs.

    The adjacency matrices follow the LiNGAM convention, where entry `(j, k)`
    represents the directed edge :math:`k \to j`. A missing edge, an extra edge, and
    a reversed edge each contribute one unit to the structural Hamming distance.

    Args:
        adjacency_matrix_true (np.typing.ArrayLike): Ground-truth weighted adjacency
            matrix.
        adjacency_matrix_pred (np.typing.ArrayLike): Estimated weighted adjacency
            matrix.

    Returns:
        int: Structural Hamming distance.

    Raises:
        ValueError: If either matrix is not square or the shapes differ.
    """
    edges_true, edges_pred = _check_adjacency_matrices(
        adjacency_matrix_true,
        adjacency_matrix_pred,
    )
    return int((_pair_states(edges_true) != _pair_states(edges_pred)).sum())


def f1_score(
    adjacency_matrix_true: np.typing.ArrayLike,
    adjacency_matrix_pred: np.typing.ArrayLike,
) -> float:
    r"""Computes the directed-edge F1 score between two adjacency matrices.

    The adjacency matrices are binarized by treating nonzero entries as edges. Diagonal
    entries are ignored, and directed edges are compared according to the LiNGAM
    convention, where entry `(j, k)` represents the edge :math:`k \to j`.

    Args:
        adjacency_matrix_true (np.typing.ArrayLike): Ground-truth weighted adjacency
            matrix.
        adjacency_matrix_pred (np.typing.ArrayLike): Estimated weighted adjacency
            matrix.

    Returns:
        float: Directed-edge F1 score.

    Raises:
        ValueError: If either matrix is not square or the shapes differ.
    """
    edges_true, edges_pred = _check_adjacency_matrices(
        adjacency_matrix_true,
        adjacency_matrix_pred,
    )
    return float(
        sklearn_f1_score(
            _off_diagonal_entries(edges_true),
            _off_diagonal_entries(edges_pred),
            zero_division=0.0,
        )
    )
