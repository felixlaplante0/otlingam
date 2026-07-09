import numpy as np
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
