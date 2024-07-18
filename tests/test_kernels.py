'''
Test of the convolutional kernels
'''

import sys
import pytest
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


def test_flat_kernel_except():
    '''
    Test the exceptions raised by the flat kernel
    '''
    clust = clue.clusterer(0.4, 5, 0.4)
    clust.read_data(clue.test_blobs(1000, 2))

    # Now we test that if we pass an incorrect set of parameters, an exception is raised
    with pytest.raises(ValueError):
        clust.choose_kernel('flat', [])
    with pytest.raises(ValueError):
        clust.choose_kernel('flat', [1., 2.])


def test_gaussian_kernel_except():
    '''
    Test the exceptions raised by the gaussian kernel
    '''
    clust = clue.clusterer(0.4, 5, 0.4)
    clust.read_data(clue.test_blobs(1000, 2))

    # Now we test that if we pass an incorrect set of parameters, an exception is raised
    with pytest.raises(ValueError):
        clust.choose_kernel('gaus', [])
    with pytest.raises(ValueError):
        clust.choose_kernel('gaus', [1.])


def test_exponential_kernel_except():
    '''
    Test the exceptions raised by the exponential kernel
    '''
    clust = clue.clusterer(0.4, 5, 0.4)
    clust.read_data(clue.test_blobs(1000, 2))

    # Now we test that if we pass an incorrect set of parameters, an exception is raised
    with pytest.raises(ValueError):
        clust.choose_kernel('exp', [])
    with pytest.raises(ValueError):
        clust.choose_kernel('exp', [1., 2., 3.])


def test_custom_kernel_except():
    '''
    Test the exceptions raised by the custom kernel
    '''
    clust = clue.clusterer(0.4, 5, 0.4)
    clust.read_data(clue.test_blobs(1000, 2))

    # Now we test that if we pass an incorrect set of parameters, an exception is raised
    with pytest.raises(ValueError):
        clust.choose_kernel('custom', [1., 2.])


def test_inexistent_kernel_except():
    '''
    Test the exceptions raised when choosing an inexistent kernel
    '''
    clust = clue.clusterer(0.4, 5, 0.4)
    clust.read_data(clue.test_blobs(1000, 2))

    # Now we test that if we pass an incorrect set of parameters, an exception is raised
    with pytest.raises(ValueError):
        clust.choose_kernel('polynomial', [1., 2.])
