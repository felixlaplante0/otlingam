"""Evaluation metrics for learned causal orders and structures."""

from __future__ import annotations

from typing import cast

import numpy as np

_MATRIX_NDIM = 2


def disorder(
    causal_order: np.typing.ArrayLike,
    adjacency_matrix: np.typing.ArrayLike,
) -> int:
    """Count the true edges reversed by an estimated causal order.

    The disorder of an estimated order is the number of true edges k -> j (i.e.
    ``adjacency_matrix[j, k] != 0``) whose endpoints appear in the wrong relative
    position, that is with k placed *after* j in the order:

        Dis(order) = #{ (k, j) : adjacency_matrix[j, k] != 0 and pos(k) > pos(j) },

    where ``pos(v)`` is the index of v in ``causal_order``. It equals zero if and only
    if ``causal_order`` is a valid topological order of the DAG encoded by
    ``adjacency_matrix`` (no effect precedes its cause). Unlike the structural Hamming
    distance, it depends only on the order, not on edge thresholding or a sparsity
    penalty, and so isolates causal-order recovery — the quantity guaranteed by the
    Wasserstein order identification theorem.

    Args:
        causal_order (np.typing.ArrayLike): Estimated order from source to sink, a
            permutation of range(d).
        adjacency_matrix (np.typing.ArrayLike): Ground-truth weighted adjacency matrix
            of shape (d, d), where entry [j, k] is the effect of k on j (k -> j).

    Returns:
        int: Number of reversed true edges (>= 0); zero iff the order is a valid
            topological order of the true DAG.

    Raises:
        ValueError: If ``causal_order`` is not a permutation of range(d) or shapes are
            inconsistent.
    """
    order = np.asarray(causal_order).ravel()
    B = cast(np.ndarray, np.asarray(adjacency_matrix, dtype=np.float64))

    if B.ndim != _MATRIX_NDIM or B.shape[0] != B.shape[1]:
        raise ValueError("adjacency_matrix must be a square (d, d) array.")
    d = B.shape[0]
    if order.shape[0] != d or sorted(order.tolist()) != list(range(d)):
        raise ValueError("causal_order must be a permutation of range(d).")

    pos = np.empty(d, dtype=np.int64)
    pos[order] = np.arange(d)

    child, parent = np.nonzero(B)
    return int(np.sum(pos[parent] > pos[child]))
