'''
Test that the equality operator for clusterer objects works correctly
'''

import sys
import pandas as pd
import pytest
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def sissa():
    '''
    Returns the dataframe containing the sissa ataset
    '''
    return pd.read_csv("../data/sissa_1000.csv")


@pytest.fixture
def toy_det():
    '''
    Returns the dataframe containing the toy detector dataset
    '''
    return pd.read_csv("../data/toyDetector_1000.csv")


def test_clusterer_equality(sissa, toy_det):
    '''
    Test the equality operator for clusterer objects
    '''
    # Sissa dataset
    clust1 = clue.clusterer(20., 10., 20.)
    clust1.read_data(sissa)
    clust1.run_clue()

    # Create a copy of the sissa lusterer to check the equality of clusterers
    clust1_copy = clue.clusterer(20., 10., 20.)
    clust1_copy.read_data(sissa)
    clust1_copy.run_clue()

    # toyDet dataset
    clust2 = clue.clusterer(5., 2.5, 5.)
    clust2.read_data(toy_det)
    clust2.run_clue()

    # Create a copy to check the equality of clusterers
    clust2_copy = clue.clusterer(5., 2.5, 5.)
    clust2_copy.read_data(toy_det)
    clust2_copy.run_clue()

    # Check equality
    assert clust1.clust_prop == clust1_copy.clust_prop
    assert clust2.clust_prop == clust2_copy.clust_prop

    # Check inequality
    assert clust1.clust_prop != clust2.clust_prop
