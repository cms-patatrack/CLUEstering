import CLUEstering as clue
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')


def test_except_1():
    clust = clue.clusterer(0.4, 5., 1.2)

    with pytest.raises(ValueError):
        clust.read_data(clue.test_blobs(n_samples=1000, n_dim=2, x_max=-3.))
    with pytest.raises(ValueError):
        clust.read_data(clue.test_blobs(n_samples=1000, n_dim=2, y_max=-2.))
    with pytest.raises(ValueError):
        clust.read_data(clue.test_blobs(
            n_samples=1000, n_dim=2, x_max=-3., y_max=-2.))


def test_except_2():
    clust = clue.clusterer(0.4, 5., 1.2)

    with pytest.raises(ValueError):
        clust.read_data(clue.test_blobs(n_samples=1000, n_dim=2, n_blobs=-2))


def test_except_3():
    clust = clue.clusterer(0.4, 5., 1.2)

    with pytest.raises(ValueError):
        clust.read_data(clue.test_blobs(n_samples=1000, n_dim=2, sigma=-2.))


def test_except_4():
    clust = clue.clusterer(0.4, 5., 1.2)

    with pytest.raises(ValueError):
        clust.read_data(clue.test_blobs(n_samples=1000, n_dim=4))


def test_successful_run():
    # Since the blobs are randomly generated, it is not possible to precisely
    # predict the result of the clustering a priory
    # So, we simply check that the clustering runs without errors
    clust = clue.clusterer(0.4, 5., 1.2)
    clust.read_data(clue.test_blobs(n_samples=1000, n_dim=2))
    clust.run_clue()
