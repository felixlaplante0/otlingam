import networkx as nx
import numpy as np
from scipy.stats import t


def _gen_dag(
    d: int,
    edge_probability: float,
    noise: np.ndarray,
    graph_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates observations from a randomly permuted linear DAG.

    Args:
        d (int): Number of variables.
        edge_probability (float): Probability of each admissible directed edge.
        noise (np.ndarray): Independent noise with shape `(n, d)`.
        graph_type (str): Random graph family. `er` denotes an Erdos--Renyi graph
            and `sf` denotes a scale-free graph.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and weighted adjacency matrix.
    """
    weights = np.random.uniform(0.5, 2, (d, d)) * np.random.choice((-1, 1), (d, d))

    if graph_type == "er":
        adjacency = np.tril(np.random.random((d, d)) < edge_probability, k=-1)
    elif graph_type == "sf":
        adjacency = np.zeros((d, d), dtype=bool)
        if d > 1 and edge_probability > 0:
            edges_per_node = max(1, round(edge_probability * (d - 1) / 2))
            graph = nx.barabasi_albert_graph(
                d,
                min(edges_per_node, d - 1),
                seed=int(np.random.randint(np.iinfo(np.int32).max)),
            )
            for parent, child in graph.edges:
                adjacency[max(parent, child), min(parent, child)] = True
    else:
        raise ValueError("graph_type must be either 'er' or 'sf'.")

    weights *= adjacency
    permutation = np.random.permutation(d)
    weights = weights[np.ix_(permutation, permutation)]
    return np.linalg.solve(np.eye(d) - weights, noise.T).T, weights


def gen_laplace(
    n: int,
    d: int,
    edge_probability: float,
    *,
    graph_type: str = "er",
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a linear DAG model with independent Laplace noise.

    Args:
        n (int): Number of observations.
        d (int): Number of variables.
        edge_probability (float): Probability of each admissible directed edge.
        graph_type (str, optional): Random graph family. Defaults to `er`.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and weighted adjacency matrix.
    """
    return _gen_dag(d, edge_probability, np.random.laplace(size=(n, d)), graph_type)


def gen_t(
    n: int,
    d: int,
    edge_probability: float,
    dfs: np.typing.ArrayLike,
    *,
    graph_type: str = "er",
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a linear DAG model with standardized Student-t noise.

    Args:
        n (int): Number of observations.
        d (int): Number of variables.
        edge_probability (float): Probability of each admissible directed edge.
        dfs (np.typing.ArrayLike): Degrees of freedom for the variables.
        graph_type (str, optional): Random graph family. Defaults to `er`.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and weighted adjacency matrix.

    Raises:
        ValueError: If `dfs` does not contain one value greater than two per variable.
    """
    dfs = np.asarray(dfs)
    if dfs.shape != (d,) or np.any(dfs <= 2):
        raise ValueError("dfs must contain one value greater than two per variable.")
    noise = np.column_stack(
        [t.rvs(df, size=n) / np.sqrt(df / (df - 2)) for df in dfs]
    )
    return _gen_dag(d, edge_probability, noise, graph_type)
