'''
Tests of the decision graph of the CLUEstering estimator
'''

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def fitted():
    '''
    Returns the estimator fitted on the sissa dataset with 1000 points
    '''
    df = pd.read_csv("../data/sissa_1000.csv")
    est = clue.CLUEstering(density_radius=20., min_density=10., outlier_distance=30.,
                           backend="cpu serial")
    return est.fit(df[["x", "y"]].to_numpy(), sample_weight=df["weight"].to_numpy())


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _scatters(ax):
    return {c.get_label(): c.get_offsets() for c in ax.collections}


def test_not_fitted():
    with pytest.raises(NotFittedError):
        clue.plot_decision_graph(clue.CLUEstering())


def test_axes(fitted):
    _, ax = plt.subplots()
    assert clue.plot_decision_graph(fitted, ax=ax) is ax
    assert clue.plot_decision_graph(fitted).axes is not None


def test_points(fitted):
    ax = clue.plot_decision_graph(fitted)
    scatters = _scatters(ax)
    # every point is drawn once
    assert sum(len(offsets) for offsets in scatters.values()) == fitted.rho_.size

    # all the seeds have an infinite delta and are drawn above all the finite ones
    seeds = scatters["seeds ($\\delta = \\infty$)"]
    assert len(seeds) == fitted.n_clusters_
    assert "seeds" not in scatters
    finite = fitted.delta_[np.isfinite(fitted.delta_)]
    assert np.all(seeds[:, 1] > finite.max())
    np.testing.assert_array_equal(np.sort(seeds[:, 0]),
                                  np.sort(fitted.rho_[fitted.seed_indices_]))


def test_thresholds(fitted):
    ax = clue.plot_decision_graph(fitted)
    lines = {line.get_label(): line for line in ax.lines}
    assert lines["min_density = 10"].get_xdata()[0] == 10.
    assert lines["seeding_distance = 20"].get_ydata()[0] == 20.
    assert lines["outlier_distance = 30"].get_ydata()[0] == 30.


def test_title(fitted):
    assert "density_radius = 20" in clue.plot_decision_graph(fitted).get_title()


def test_log_rho(fitted):
    assert clue.plot_decision_graph(fitted, log_rho=True).get_xscale() == "log"


if __name__ == "__main__":
    df = pd.read_csv("../data/sissa_1000.csv")
    est = clue.CLUEstering(density_radius=20., min_density=10., outlier_distance=30.,
                           backend="cpu serial")
    est.fit(df[["x", "y"]].to_numpy(), sample_weight=df["weight"].to_numpy())
    ax = clue.plot_decision_graph(est)
    ax.figure.savefig("decision_graph.png", dpi=300)
