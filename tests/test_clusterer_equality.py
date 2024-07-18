'''
Test that the equality operator for clusterer objects works correctly
'''

import sys
import pandas as pd
import pytest
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def moons():
    '''
    Returns the dataframe containing the moon dataset
    '''
    return pd.read_csv("./test_datasets/moons.csv")


@pytest.fixture
def circles():
    '''
    Returns the dataframe containing the circle dataset
    '''
    return pd.read_csv("./test_datasets/circles.csv")


def test_clusterer_equality(moons, circles):
    '''
    Test the equality operator for clusterer objects
    '''
    # Moons dataset
    clust1 = clue.clusterer(0.5, 5, 0.5)
    clust1.read_data(moons)
    clust1.run_clue()

    # Create a copy of the moons clusterer to check the equality of clusterers
    clust1_copy = clue.clusterer(0.5, 5, 0.5)
    clust1_copy.read_data(moons)
    clust1_copy.run_clue()

    # Circles dataset
    clust2 = clue.clusterer(0.9, 5, 0.9)
    clust2.read_data(circles)
    clust2.run_clue()

    # Create a copy of the circles clusterer to check the equality of clusterers
    clust2_copy = clue.clusterer(0.9, 5, 0.9)
    clust2_copy.read_data(circles)
    clust2_copy.run_clue()

    # Check equality
    assert clust1.clust_prop == clust1_copy.clust_prop
    assert clust2.clust_prop == clust2_copy.clust_prop

    # Check inequality
    assert clust1.clust_prop != clust2.clust_prop
