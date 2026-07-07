"""Tests for otlingam estimator workflows."""

import numpy as np
import pytest
from sklearn.base import clone

from otlingam import ExhaustiveOTLiNGAM, GreedyOTLiNGAM, OTICALiNGAM

from ._utils import linear_dag


@pytest.mark.parametrize("estimator_class", [ExhaustiveOTLiNGAM, GreedyOTLiNGAM])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_score_estimator_fit(estimator_class, fit_intercept):
    """Exercises the public fit workflow for score-based estimators."""
    X, adjacency_matrix = linear_dag()
    estimator = estimator_class(fit_intercept=fit_intercept)

    assert clone(estimator).get_params() == {"fit_intercept": fit_intercept}
    assert estimator.fit(X) is estimator
    assert sorted(estimator.causal_order_) == [0, 1, 2]
    assert estimator.adjacency_matrix_.shape == adjacency_matrix.shape
    assert np.isfinite(estimator.score_)

    if fit_intercept:
        assert estimator.intercept_.shape == (3,)
    else:
        assert "intercept_" not in estimator.__dict__


def test_greedy_constant_residuals():
    """Checks that degenerate residuals produce a clear error."""
    X = np.ones((4, 2))

    with pytest.raises(ValueError, match="constant residual"):
        GreedyOTLiNGAM().fit(X)


def test_oticalingam_fit():
    """Exercises the OTICA-backed LiNGAM fit workflow."""
    X, _ = linear_dag()
    estimator = OTICALiNGAM(random_state=42, max_iter=1)

    assert estimator.fit(X, y=None) is estimator
    assert estimator.adjacency_matrix_.shape == (3, 3)
    assert estimator.intercept_.shape == (3,)
    assert sorted(estimator.causal_order_) == [0, 1, 2]
    assert np.isfinite(estimator.adjacency_matrix_).all()
    assert np.isfinite(estimator.intercept_).all()
