import numpy as np
from scipy.stats import t


def _gen_dag(
    n: int,
    d: int,
    edge_probability: float,
    noise: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates observations from a randomly permuted linear DAG.

    Args:
        n (int): Number of observations.
        d (int): Number of variables.
        edge_probability (float): Probability of each admissible directed edge.
        noise (np.ndarray): Independent noise with shape `(n, d)`.
        rng (np.random.Generator): Random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and weighted adjacency matrix.
    """
    weights = rng.uniform(0.5, 2, (d, d)) * rng.choice((-1, 1), (d, d))
    weights *= np.tril(rng.random((d, d)) < edge_probability, k=-1)
    permutation = rng.permutation(d)
    weights = weights[np.ix_(permutation, permutation)]
    return np.linalg.solve(np.eye(d) - weights, noise.T).T, weights


def gen_laplace(
    n: int,
    d: int,
    edge_probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a linear DAG model with independent Laplace noise.

    Args:
        n (int): Number of observations.
        d (int): Number of variables.
        edge_probability (float): Probability of each admissible directed edge.
        rng (np.random.Generator): Random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and weighted adjacency matrix.
    """
    return _gen_dag(n, d, edge_probability, rng.laplace(size=(n, d)), rng)


def gen_t(
    n: int,
    d: int,
    edge_probability: float,
    dfs: np.typing.ArrayLike,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a linear DAG model with standardized Student-t noise.

    Args:
        n (int): Number of observations.
        d (int): Number of variables.
        edge_probability (float): Probability of each admissible directed edge.
        dfs (np.typing.ArrayLike): Degrees of freedom for the variables.
        rng (np.random.Generator): Random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and weighted adjacency matrix.

    Raises:
        ValueError: If `dfs` does not contain one value greater than two per variable.
    """
    dfs = np.asarray(dfs)
    if dfs.shape != (d,) or np.any(dfs <= 2):
        raise ValueError("dfs must contain one value greater than two per variable.")
    noise = np.column_stack(
        [t.rvs(df, size=n, random_state=rng) / np.sqrt(df / (df - 2)) for df in dfs]
    )
    return _gen_dag(n, d, edge_probability, noise, rng)
