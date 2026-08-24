"""Tests for exhaustive causal discovery."""

import numpy as np
import pytest

from otlingam import ExhaustiveOTLiNGAM
from otlingam.models._exhaustive import _MAX_DP_VARIABLES

from ._utils import linear_dag


def test_fit():
    """Fits the exhaustive estimator on a small linear DAG."""
    X, _ = linear_dag()
    model = ExhaustiveOTLiNGAM().fit(X)

    assert np.isfinite(model.score_)
    assert sorted(model.causal_order_) == [0, 1, 2]


def test_singular_residuals():
    """Handles singular regression systems without crashing."""
    model = ExhaustiveOTLiNGAM().fit(np.ones((4, 2)))

    assert not np.isfinite(model.score_)


def test_variable_limit():
    """Rejects inputs that would exceed the subset-state limit."""
    X = np.ones((2, _MAX_DP_VARIABLES + 1))

    with pytest.raises(ValueError, match="at most"):
        ExhaustiveOTLiNGAM().fit(X)
