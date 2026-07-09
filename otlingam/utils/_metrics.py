import numpy as np
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.utils.validation import check_array, column_or_1d  # type: ignore


def _check_adjacency_matrices(
    true_adjacency_matrix: np.typing.ArrayLike,
    pred_adjacency_matrix: np.typing.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Validates and binarizes a pair of adjacency matrices.

    Args:
        true_adjacency_matrix (np.typing.ArrayLike): Ground-truth weighted adjacency
            matrix.
        pred_adjacency_matrix (np.typing.ArrayLike): Estimated weighted adjacency
            matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: Binary true and estimated adjacency matrices.

    Raises:
        ValueError: If either matrix is not square or the shapes differ.
    """
    true_edges = check_array(true_adjacency_matrix) != 0.0
    pred_edges = check_array(pred_adjacency_matrix) != 0.0

    if true_edges.shape[0] != true_edges.shape[1]:
        raise ValueError(
            "true_adjacency_matrix must be a square array, "
            f"got shape {true_edges.shape}."
        )
    if pred_edges.shape[0] != pred_edges.shape[1]:
        raise ValueError(
            "pred_adjacency_matrix must be a square array, "
            f"got shape {pred_edges.shape}."
        )
    if true_edges.shape != pred_edges.shape:
        raise ValueError(
            "true_adjacency_matrix and pred_adjacency_matrix must have the same "
            f"shape, got {true_edges.shape} and {pred_edges.shape}."
        )

    return true_edges, pred_edges


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
    true_adjacency_matrix: np.typing.ArrayLike,
    pred_causal_order: np.typing.ArrayLike,
) -> int:
    r"""Counts true edges reversed by a causal order.

    Let :math:`\hat{\sigma}` be the estimated order. The disorder is given by

    .. math::

        \begin{aligned}
        \mathrm{dis}(\hat{\sigma}) &= \#\left\{ (k, j) : B^\star_{jk} \neq 0, \\
        &\quad \hat{\sigma}^{-1}(k) > \hat{\sigma}^{-1}(j) \right\}.
        \end{aligned}

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


def shd(
    true_adjacency_matrix: np.typing.ArrayLike,
    pred_adjacency_matrix: np.typing.ArrayLike,
) -> int:
    r"""Computes structural Hamming distance between two directed graphs.

    The adjacency matrices follow the LiNGAM convention, where entry `(j, k)`
    represents the directed edge :math:`k \to j`. A missing edge, an extra edge, and
    a reversed edge each contribute one unit to the structural Hamming distance.

    Args:
        true_adjacency_matrix (np.typing.ArrayLike): Ground-truth weighted adjacency
            matrix.
        pred_adjacency_matrix (np.typing.ArrayLike): Estimated weighted adjacency
            matrix.

    Returns:
        int: Structural Hamming distance.

    Raises:
        ValueError: If either matrix is not square or the shapes differ.
    """
    true_edges, pred_edges = _check_adjacency_matrices(
        true_adjacency_matrix,
        pred_adjacency_matrix,
    )
    return int((_pair_states(true_edges) != _pair_states(pred_edges)).sum())


def f1_score(
    true_adjacency_matrix: np.typing.ArrayLike,
    pred_adjacency_matrix: np.typing.ArrayLike,
) -> float:
    r"""Computes the directed-edge F1 score between two adjacency matrices.

    The adjacency matrices are binarized by treating nonzero entries as edges. Diagonal
    entries are ignored, and directed edges are compared according to the LiNGAM
    convention, where entry `(j, k)` represents the edge :math:`k \to j`.

    Args:
        true_adjacency_matrix (np.typing.ArrayLike): Ground-truth weighted adjacency
            matrix.
        pred_adjacency_matrix (np.typing.ArrayLike): Estimated weighted adjacency
            matrix.

    Returns:
        float: Directed-edge F1 score.

    Raises:
        ValueError: If either matrix is not square or the shapes differ.
    """
    true_edges, pred_edges = _check_adjacency_matrices(
        true_adjacency_matrix,
        pred_adjacency_matrix,
    )
    return float(
        sklearn_f1_score(
            _off_diagonal_entries(true_edges),
            _off_diagonal_entries(pred_edges),
            zero_division=0.0,
        )
    )
