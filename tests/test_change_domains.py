'''
Testing the function for changing the domain ranges, using the blob dataset as a reference
'''

from math import pi
import numpy as np
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def blob():
    '''
    Returns the dataframe containing the blob dataset
    '''
    csv_file = './test_datasets/blob.csv'
    return csv_file


def test_default_domains(blob):
    '''
    Check the values of the default domain ranges
    '''
    clust = clue.clusterer(0.5, 5., 1.2)
    clust.read_data(blob)

    # Check that the domains extremes of the two coordinates are the default ones
    # equal to +/- the numerical limit of a float
    assert clust.clust_data.domain_ranges[0].min == -3.4028234663852886e+38
    assert clust.clust_data.domain_ranges[0].max == 3.4028234663852886e+38
    assert clust.clust_data.domain_ranges[1].min == -3.4028234663852886e+38
    assert clust.clust_data.domain_ranges[1].max == 3.4028234663852886e+38


def test_change_domains_1():
    '''
    Check the renormalization for uniform data
    '''
    # We generate data with zero mean and standard deviation, so that the
    # domain extremes are not normalized by the standard scaler
    x0 = np.zeros(shape=5)
    x1 = np.zeros(shape=5)
    weight = np.full(shape=5, fill_value=1)
    data = {'x0': x0, 'x1': x1, 'weight': weight}

    clust = clue.clusterer(0.5, 5., 1.2)
    clust.read_data(data)

    # Check that the original domains are the numerical limits of float
    # like in the previous test case
    assert clust.clust_data.domain_ranges[0].min == -3.4028234663852886e+38
    assert clust.clust_data.domain_ranges[0].max == 3.4028234663852886e+38
    assert clust.clust_data.domain_ranges[1].min == -3.4028234663852886e+38
    assert clust.clust_data.domain_ranges[1].max == 3.4028234663852886e+38

    # Change the domains
    clust.change_domains(x0=(0., 2.), x1=(-pi, pi))

    # Check that the new domains are (0, 2) and (-pi, pi)
    assert clust.clust_data.domain_ranges[0].min == 0.
    assert clust.clust_data.domain_ranges[0].max == 2.
    assert clust.clust_data.domain_ranges[1].min == pytest.approx(
        -pi, 0.0000001)
    assert clust.clust_data.domain_ranges[1].max == pytest.approx(
        pi, 0.0000001)


def test_change_domains_2():
    '''
    Check the renormalization for non-uniform data
    '''
    # We generate data with non-zero mean and standard deviation, and we check
    # that the domain exctremes are re-calculated as expected by the scaler
    x0 = np.arange(0, 5)
    x1 = np.arange(0, 5)
    weight = np.full(shape=5, fill_value=1)
    data = {'x0': x0, 'x1': x1, 'weight': weight}

    clust = clue.clusterer(0.5, 5., 1.2)
    clust.read_data(data)

    # Check that the original domains are the numerical limits of float
    # like in the previous test case
    assert clust.clust_data.domain_ranges[0].min == -3.4028234663852886e+38
    assert clust.clust_data.domain_ranges[0].max == 3.4028234663852886e+38
    assert clust.clust_data.domain_ranges[1].min == -3.4028234663852886e+38
    assert clust.clust_data.domain_ranges[1].max == 3.4028234663852886e+38

    # Change the domains
    clust.change_domains(x0=(0., 2.), x1=(-pi, pi))

    # Check that the new domains are (0, 2) and (-pi, pi)
    assert clust.clust_data.domain_ranges[0].min == pytest.approx(-1.41, 0.01)
    assert clust.clust_data.domain_ranges[0].max == 0.
    assert clust.clust_data.domain_ranges[1].min == pytest.approx(
        -3.6356550, 0.0000001)
    assert clust.clust_data.domain_ranges[1].max == pytest.approx(
        0.8072279, 0.0000001)
