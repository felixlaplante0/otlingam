"""Tests for otlingam estimator workflows."""

import numpy as np
import pytest
from sklearn.base import clone

import otlingam._ica as ica_module
from otlingam import ExhaustiveOTLiNGAM, GreedyOTLiNGAM, OTICALiNGAM

from ._utils import linear_dag


@pytest.mark.parametrize("estimator_class", [ExhaustiveOTLiNGAM, GreedyOTLiNGAM])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_score_estimators_fit_public_workflow(estimator_class, fit_intercept):
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


def test_greedy_rejects_constant_residuals():
    """Checks that degenerate residuals produce a clear error."""
    X = np.ones((4, 2))

    with pytest.raises(ValueError, match="constant residual"):
        GreedyOTLiNGAM().fit(X)


def test_oticalingam_fit_uses_otica_components(monkeypatch):
    """Exercises the OTICA-backed LiNGAM fit workflow with a deterministic double."""
    X, _ = linear_dag()

    class FakeOTICA:
        """Provides deterministic OTICA attributes used by `OTICALiNGAM.fit`."""

        def __init__(self, *, max_iter, random_state):
            self.max_iter = max_iter
            self.random_state = random_state

        def fit(self, X):
            self.components_ = np.full((X.shape[1], X.shape[1]), 0.1)
            np.fill_diagonal(self.components_, 2.0)
            self.mean_ = X.mean(axis=0)

            return self

    def fake_estimate_causal_order(_self, B_estimate):
        assert B_estimate.shape == (3, 3)

        return [0, 1, 2]

    def fake_estimate_adjacency_matrix(self, X):
        self._adjacency_matrix = np.zeros((X.shape[1], X.shape[1]))

    monkeypatch.setattr(ica_module, "OTICA", FakeOTICA)
    monkeypatch.setattr(
        OTICALiNGAM,
        "_estimate_causal_order",
        fake_estimate_causal_order,
    )
    monkeypatch.setattr(
        OTICALiNGAM,
        "_estimate_adjacency_matrix",
        fake_estimate_adjacency_matrix,
    )

    estimator = OTICALiNGAM(random_state=42, max_iter=5)

    assert estimator.fit(X) is estimator
    assert estimator.causal_order_ == [0, 1, 2]
    assert estimator.adjacency_matrix_.shape == (3, 3)
    assert np.allclose(estimator.intercept_, X.mean(axis=0))
