import numpy as np
import pytest

from otlingam import disorder
from otica._utils import gauss_quantiles

from ._utils import linear_dag


def test_disorder_count():
    """Counts true edges that conflict with the supplied order."""
    _, adjacency_matrix = linear_dag()

    assert disorder([0, 1, 2], adjacency_matrix) == 0
    assert disorder([2, 1, 0], adjacency_matrix) == np.count_nonzero(adjacency_matrix)


def test_disorder_validation():
    """Checks validation for non-square matrices and non-permutation orders."""
    with pytest.raises(ValueError, match="square array"):
        disorder([0, 1], np.ones((2, 3)))

    with pytest.raises(ValueError, match="permutation"):
        disorder([0, 0], np.eye(2))


def test_gauss_quantiles():
    """Checks basic Gaussian rank-statistic properties."""
    quantiles = gauss_quantiles(6)

    assert quantiles.shape == (6,)
    assert np.all(np.diff(quantiles) > 0.0)
    assert np.isclose(quantiles.mean(), 0.0)
    
