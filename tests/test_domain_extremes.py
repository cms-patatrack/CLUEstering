'''
Test that points at opposite extremes of a finite domain are adjacent
'''

from math import pi
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def opposite_angles():
    '''
    Returns a dataset with points distributed at opposite sides of a finite range
    '''
    return pd.read_csv("./test_datasets/opposite_angles.csv")


def test_opposite_angles(opposite_angles):
    '''
    Test the clustering of points at opposite angles
    '''
    # Test points with angles distributed at opposite extremes of the domain
    # This test assures that the code works for data with periodic coordinates
    clust = clue.clusterer(0.1, 1, 1.1)
    clust.read_data(opposite_angles, x1=(-pi, pi))
    clust.run_clue()

    # We just check that the number of clusters is one
    # If it is, the code recognizes points at the opposite extremes of the domain
    # as very near each other
    assert clust.clust_prop.n_clusters == 1
