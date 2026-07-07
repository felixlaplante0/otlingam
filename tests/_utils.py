import numpy as np


def linear_dag() -> tuple[np.ndarray, np.ndarray]:
    """Creates deterministic observations from a small linear DAG.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and weighted adjacency matrix.
    """
    noise = np.array(
        [
            [-1.5, -0.8, -0.2],
            [-1.0, -0.1, 0.7],
            [-0.4, 0.6, -1.1],
            [0.2, 1.1, 0.4],
            [0.9, -1.3, 1.2],
            [1.4, 0.5, -0.6],
        ]
    )
    adjacency_matrix = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [-0.4, 0.6, 0.0],
        ]
    )
    X = noise @ np.linalg.inv(np.eye(3) - adjacency_matrix).T

    return X, adjacency_matrix
