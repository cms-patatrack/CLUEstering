'''
Test the test_blobs function, which produces a set of gaussianely distributed blobs
'''

import sys
import pytest
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue



def test_except_1():
    '''
    Test exception of test_blobs for negative number of blobs
    '''
    clust = clue.clusterer(0.4, 5., 1.2)

    with pytest.raises(ValueError):
        clust.read_data(clue.test_blobs(n_samples=1000, n_dim=2, n_blobs=-2))


def test_except_2():
    '''
    Test exception of test_blobs for negative standard deviations
    '''
    clust = clue.clusterer(0.4, 5., 1.2)

    with pytest.raises(ValueError):
        clust.read_data(clue.test_blobs(n_samples=1000, n_dim=2, sigma=-2.))


def test_except_3():
    '''
    Test exception of test_blobs for too high dimensions
    '''
    clust = clue.clusterer(0.4, 5., 1.2)

    with pytest.raises(ValueError):
        clust.read_data(clue.test_blobs(n_samples=1000, n_dim=4))


def test_successful_run():
    '''
    Test correct use of test_blobs
    '''
    # Since the blobs are randomly generated, it is not possible to precisely
    # predict the result of the clustering a priory
    # So, we simply check that the clustering runs without errors
    clust = clue.clusterer(0.4, 5., 1.2)
    clust.read_data(clue.test_blobs(n_samples=1000, n_dim=2))
    clust.run_clue()
