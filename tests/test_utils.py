import numpy as np
import pytest

from otlingam.models._utils import gauss_quantiles
from otlingam.utils import disorder, f1_score, shd

from ._utils import linear_dag


def test_disorder_count():
    """Counts true edges that conflict with the supplied order."""
    _, adjacency_matrix = linear_dag()

    assert disorder(adjacency_matrix, [0, 1, 2]) == 0
    assert disorder(adjacency_matrix, [2, 1, 0]) == np.count_nonzero(adjacency_matrix)


def test_disorder_validation():
    """Checks validation for non-square matrices and non-permutation orders."""
    with pytest.raises(ValueError, match="square array"):
        disorder(np.ones((2, 3)), [0, 1])

    with pytest.raises(ValueError, match="permutation"):
        disorder(np.eye(2), [0, 0])


def test_shd():
    """Counts missing, extra, and reversed directed edges."""
    adjacency_matrix_true = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    adjacency_matrix_pred = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert shd(adjacency_matrix_true, adjacency_matrix_pred) == 2  # noqa: PLR2004


def test_f1_score():
    """Computes the directed-edge F1 score from nonzero entries."""
    adjacency_matrix_true = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    adjacency_matrix_pred = np.array(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    assert f1_score(adjacency_matrix_true, adjacency_matrix_pred) == pytest.approx(0.5)
    assert f1_score(np.zeros((2, 2)), np.eye(2)) == 0.0


@pytest.mark.parametrize("metric", [shd, f1_score])
def test_graph_metric_validation(metric):
    """Checks graph-metric validation for incompatible adjacency matrices."""
    with pytest.raises(ValueError, match="square array"):
        metric(np.ones((2, 3)), np.eye(2))

    with pytest.raises(ValueError, match="same shape"):
        metric(np.eye(2), np.eye(3))


def test_gauss_quantiles():
    """Checks basic Gaussian rank-statistic properties."""
    quantiles = gauss_quantiles(6)

    assert quantiles.shape == (6,)
    assert np.all(np.diff(quantiles) > 0.0)
    assert np.isclose(quantiles.mean(), 0.0)
