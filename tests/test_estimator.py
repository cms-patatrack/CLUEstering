'''
Tests of the scikit-learn compatible CLUEstering estimator
'''

import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def sissa():
    '''
    Returns the coordinates and weights of the sissa dataset with 1000 points
    '''
    df = pd.read_csv("../data/sissa_1000.csv")
    return df[["x", "y"]].to_numpy(), df["weight"].to_numpy()


@parametrize_with_checks(
    [clue.CLUEstering(density_radius=0.5, min_density=1.0, backend="cpu serial"),
     clue.CLUEstering(density_radius=0.5, min_density=1.0, merge_threshold=0.5,
                      backend="cpu serial")],
    expected_failed_checks=lambda est: {
        # k copies of a point have the same density and compete as nearest highers, so they
        # are not equivalent to a single point with weight k. The check also uses 30 features.
        "check_sample_weight_equivalence_on_dense_data":
            "integer weights are not equivalent to repeated points",
        # these checks use 5 or 10 features, the bindings are compiled for up to 4
        "check_estimators_dtypes": "more than 4 features",
        "check_dtype_object": "more than 4 features",
        "check_fit2d_1sample": "more than 4 features",
    })
def test_sklearn_compatible(estimator, check):
    check(estimator)


def test_same_partition_as_clusterer(sissa):
    X, weights = sissa
    old = clue.clusterer(20., 10., 20.)
    old.read_data(pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "weight": weights}))
    old.run_clue(backend="cpu serial")

    est = clue.CLUEstering(density_radius=20., min_density=10., backend="cpu serial")
    labels = est.fit_predict(X, sample_weight=weights)
    assert est.n_clusters_ == old.n_clusters
    assert adjusted_rand_score(labels, old.cluster_ids) == 1.0


@pytest.mark.parametrize("backend", clue.backends)
def test_labels_independent_of_backend(sissa, backend):
    X, weights = sissa
    reference = clue.CLUEstering(20., 10., backend="cpu serial").fit_predict(X, sample_weight=weights)
    labels = clue.CLUEstering(20., 10., backend=backend).fit_predict(X, sample_weight=weights)
    np.testing.assert_array_equal(labels, reference)


def test_params_are_not_resolved_in_init():
    est = clue.CLUEstering(density_radius=2.)
    assert est.outlier_distance is None
    assert clone(est).get_params() == est.get_params()
    est.set_params(min_density=5.)
    assert est.min_density == 5. and est.density_radius == 2.


def test_pipeline(sissa):
    X, weights = sissa
    pipe = make_pipeline(StandardScaler(),
                         clue.CLUEstering(density_radius=0.1, min_density=10., backend="cpu serial"))
    labels = pipe.fit_predict(X, cluestering__sample_weight=weights)
    assert labels.shape == (X.shape[0],)


def test_batched(sissa):
    X, weights = sissa
    est = clue.CLUEstering(20., 10., backend="cpu serial")
    single = est.fit_predict(X, sample_weight=weights)
    both = est.fit_predict(np.vstack([X, X]), sample_weight=np.concatenate([weights, weights]),
                           sample_sizes=[len(X), len(X)])
    # each copy is clustered independently, with clusters numbered across the batch
    assert est.n_clusters_ == 2 * single.max() + 2
    assert adjusted_rand_score(both[:len(X)], single) == 1.0
    assert adjusted_rand_score(both[len(X):], single) == 1.0
    assert not set(both[:len(X)][both[:len(X)] >= 0]) & set(both[len(X):][both[len(X):] >= 0])


@pytest.mark.parametrize("params, message", [
    ({"kernel": "box"}, "Invalid kernel"),
    ({"kernel": "gaussian"}, "requires 2 parameter"),
    ({"metric": "minkowski"}, "Invalid metric"),
    ({"metric": "periodic_euclidean"}, "requires metric_params"),
    ({"metric_params": {"weights": [1.]}}, "one value per feature"),
    ({"wrapped": [1]}, "one flag per feature"),
    ({"density_radius": -1.}, "must be positive"),
    ({"backend": "quantum"}, "Invalid backend"),
])
def test_invalid_params(sissa, params, message):
    X, _ = sissa
    with pytest.raises(ValueError, match=message):
        clue.CLUEstering(backend="cpu serial").set_params(**params).fit(X)


def test_too_many_features():
    with pytest.raises(ValueError, match="up to 4 features"):
        clue.CLUEstering(backend="cpu serial").fit(np.zeros((5, 5)))


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_layouts(sissa, order, dtype):
    X, weights = sissa
    reference = clue.CLUEstering(20., 10., backend="cpu serial").fit_predict(X, sample_weight=weights)
    labels = clue.CLUEstering(20., 10., backend="cpu serial").fit_predict(
        np.asarray(X, dtype=dtype, order=order), sample_weight=weights.astype(dtype))
    np.testing.assert_array_equal(labels, reference)


@pytest.mark.parametrize("batched", [False, True])
def test_details(sissa, batched):
    X, weights = sissa
    size = X.shape[0]
    sample_sizes = None
    if batched:
        X, weights = np.vstack([X, X]), np.concatenate([weights, weights])
        sample_sizes = [size, size]
    est = clue.CLUEstering(20., 10., backend="cpu serial")
    est.fit(X, sample_weight=weights, sample_sizes=sample_sizes)
    n = X.shape[0]
    assert est.rho_.shape == est.delta_.shape == est.nearest_higher_.shape == (n,)

    # one seed per cluster, without a nearest higher
    np.testing.assert_array_equal(est.labels_[est.seed_indices_], np.arange(est.n_clusters_))
    assert np.all(est.nearest_higher_[est.seed_indices_] == -1)

    # delta is infinite exactly for the points without a nearest higher
    no_nh = est.nearest_higher_ == -1
    assert np.all(np.isinf(est.delta_[no_nh]))
    assert np.all(est.delta_[~no_nh] <= est.get_params()["density_radius"])

    # the nearest higher has a higher density and belongs to the same cluster
    i = np.flatnonzero(~no_nh)
    nh = est.nearest_higher_[i]
    assert np.all(est.rho_[nh] >= est.rho_[i])
    np.testing.assert_array_equal(est.labels_[nh], est.labels_[i])

    if batched:
        # the nearest highers do not cross the samples
        assert np.all((i < size) == (nh < size))


def test_details_independent_of_backend(sissa):
    X, weights = sissa
    reference = clue.CLUEstering(20., 10., backend="cpu serial").fit(X, sample_weight=weights)
    for backend in clue.backends:
        est = clue.CLUEstering(20., 10., backend=backend).fit(X, sample_weight=weights)
        np.testing.assert_allclose(est.rho_, reference.rho_, rtol=1e-5)
        np.testing.assert_array_equal(est.seed_indices_, reference.seed_indices_)


def test_merge_disabled_by_default(sissa):
    X, weights = sissa
    plain = clue.CLUEstering(20., 10., backend="cpu serial").fit(X, sample_weight=weights)
    est = clue.CLUEstering(20., 10., merge_threshold=None, backend="cpu serial")
    est.fit(X, sample_weight=weights)
    np.testing.assert_array_equal(est.labels_, plain.labels_)
    np.testing.assert_array_equal(est.seed_indices_, plain.seed_indices_)


@pytest.mark.parametrize("threshold", [1.0, 0.5, 0.1])
def test_merge_coarsens_the_clusters(sissa, threshold):
    X, weights = sissa
    plain = clue.CLUEstering(20., 10., backend="cpu serial").fit(X, sample_weight=weights)
    est = clue.CLUEstering(20., 10., merge_threshold=threshold, backend="cpu serial")
    est.fit(X, sample_weight=weights)

    # the outliers do not change, and each plain cluster ends up entirely in one merged cluster
    np.testing.assert_array_equal(est.labels_ == -1, plain.labels_ == -1)
    clustered = plain.labels_ >= 0
    pairs = np.unique(np.stack([plain.labels_[clustered], est.labels_[clustered]]), axis=1)
    assert pairs.shape[1] == plain.n_clusters_
    assert est.n_clusters_ == np.unique(pairs[1]).size <= plain.n_clusters_

    # one seed per merged cluster, the plain seed with the highest density
    np.testing.assert_array_equal(est.labels_[est.seed_indices_], np.arange(est.n_clusters_))
    assert np.all(est.nearest_higher_[est.seed_indices_] == -1)
    for k, seed in enumerate(est.seed_indices_):
        plain_seeds = plain.seed_indices_[est.labels_[plain.seed_indices_] == k]
        assert est.rho_[seed] == est.rho_[plain_seeds].max()


def test_lower_merge_threshold_merges_more(sissa):
    X, weights = sissa
    n_clusters = [clue.CLUEstering(20., 10., merge_threshold=threshold, backend="cpu serial")
                  .fit(X, sample_weight=weights).n_clusters_
                  for threshold in (1.0, 0.5, 0.1)]
    assert n_clusters == sorted(n_clusters, reverse=True)
    # the dataset has clusters separated by shallow valleys, so the test is not trivial
    assert n_clusters[-1] < n_clusters[0]


@pytest.mark.parametrize("backend", clue.backends)
def test_merge_independent_of_backend(sissa, backend):
    X, weights = sissa
    reference = clue.CLUEstering(20., 10., merge_threshold=0.1, backend="cpu serial")
    reference.fit(X, sample_weight=weights)
    est = clue.CLUEstering(20., 10., merge_threshold=0.1, backend=backend)
    est.fit(X, sample_weight=weights)
    np.testing.assert_array_equal(est.labels_, reference.labels_)
    np.testing.assert_array_equal(est.seed_indices_, reference.seed_indices_)


@pytest.mark.parametrize("threshold, error", [(0., ValueError), (-0.5, ValueError),
                                              (1.5, ValueError), ("0.5", TypeError)])
def test_invalid_merge_threshold(sissa, threshold, error):
    X, weights = sissa
    with pytest.raises(error, match="merge_threshold"):
        clue.CLUEstering(20., 10., merge_threshold=threshold).fit(X, sample_weight=weights)


def test_merge_not_supported_with_sample_sizes(sissa):
    X, weights = sissa
    size = X.shape[0]
    est = clue.CLUEstering(20., 10., merge_threshold=0.5, backend="cpu serial")
    with pytest.raises(ValueError, match="sample_sizes"):
        est.fit(np.vstack([X, X]), sample_weight=np.concatenate([weights, weights]),
                sample_sizes=[size, size])
