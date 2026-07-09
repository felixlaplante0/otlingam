import numpy as np
from lingam.utils import predict_adaptive_lasso  # type: ignore
from scipy.special import ndtri  # type: ignore


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
