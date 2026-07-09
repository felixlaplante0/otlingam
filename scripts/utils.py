from typing import Self

import dagma.linear
import networkx as nx
import numpy as np
from dagma.utils import simulate_dag, simulate_parameter
from scipy.stats import t
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data
from tqdm.auto import tqdm

# Silence DAGMA
dagma.linear.tqdm = lambda *args, **kwargs: tqdm(*args, disable=True, **kwargs)


class DAGMA(BaseEstimator):
    """Wraps linear DAGMA with the adjacency convention used by LiNGAM.

    DAGMA represents an edge from variable ``i`` to variable ``j`` by entry ``(i, j)``.
    LiNGAM uses entry ``(j, i)`` for the same edge. This estimator transposes DAGMA's
    result and exposes both ``adjacency_matrix_`` and ``causal_order_`` after fitting.

    Attributes:
        adjacency_matrix_ (np.ndarray): Estimated weighted adjacency matrix in the
            LiNGAM convention.
        causal_order_ (list[int]): Estimated topological ordering of the variables.
    """

    def fit(self, X: object, y: object | None = None) -> Self:
        """Fits linear DAGMA to the observations.

        Args:
            X (object): Observations with samples as rows and variables as columns.
            y (object | None, optional): Ignored target values. Defaults to ``None``.

        Returns:
            Self: The fitted estimator.
        """
        X = validate_data(self, X)
        weights = dagma.linear.DagmaLinear(loss_type="l2").fit(X)
        graph = nx.from_numpy_array(weights, create_using=nx.DiGraph)
        self.adjacency_matrix_ = weights.T
        self.causal_order_ = list(nx.topological_sort(graph))

        return self


def _gen_dag(
    d: int,
    edges_per_node: int,
    noise: np.ndarray,
    graph_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates observations from a randomly permuted linear DAG.

    Args:
        d (int): Number of variables.
        edges_per_node (int): Target number of edges per node. For an
            Erd\"os--R\'enyi graph, the expected total number of edges is
            ``edges_per_node * d``, subject to the maximum number of DAG edges.
            For a scale-free graph, this is the Barab\'asi--Albert attachment
            parameter, capped at ``d - 1``.
        noise (np.ndarray): Independent noise with shape ``(n, d)``.
        graph_type (str): Random graph family. ``er`` denotes an Erdos--Renyi graph
            and ``sf`` denotes a scale-free graph.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and weighted adjacency matrix.
    """
    if graph_type == "er":
        edge_count = min(edges_per_node * d, d * (d - 1) // 2)
    else:
        edge_count = min(edges_per_node, d - 1) * d
    adjacency = simulate_dag(d, edge_count, graph_type.upper())
    weights = simulate_parameter(adjacency).T
    return np.linalg.solve(np.eye(d) - weights, noise.T).T, weights


def gen_laplace(
    n: int,
    d: int,
    edges_per_node: int,
    *,
    graph_type: str = "er",
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a linear DAG model with heterogeneously scaled Laplace noise.

    Args:
        n (int): Number of observations.
        d (int): Number of variables.
        edges_per_node (int): Target number of edges per node.
        graph_type (str, optional): Random graph family. Defaults to ``er``.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and weighted adjacency matrix.
    """
    scales = np.random.uniform(0.5, 2.0, size=d)
    noise = np.random.laplace(size=(n, d)) * scales

    return _gen_dag(d, edges_per_node, noise, graph_type)


def gen_t(
    n: int,
    d: int,
    edges_per_node: int,
    dfs: np.typing.ArrayLike,
    *,
    graph_type: str = "er",
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a linear DAG model with heterogeneously scaled Student-t noise.

    Args:
        n (int): Number of observations.
        d (int): Number of variables.
        edges_per_node (int): Target number of edges per node.
        dfs (np.typing.ArrayLike): Degrees of freedom for the variables.
        graph_type (str, optional): Random graph family. Defaults to ``er``.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and weighted adjacency matrix.

    """
    dfs = np.asarray(dfs)
    scales = np.random.uniform(0.5, 2.0, size=d)
    noise = (
        np.column_stack([t.rvs(df, size=n) / np.sqrt(df / (df - 2)) for df in dfs])
        * scales
    )

    return _gen_dag(d, edges_per_node, noise, graph_type)
