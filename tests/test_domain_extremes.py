import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue
from math import pi

@pytest.fixture
def opposite_angles():
    return pd.read_csv("./test_datasets/opposite_angles.csv")

def test_opposite_angles(opposite_angles):
    # Test points with angles distributed at opposite extremes of the domain
    # This test assures that the code works for data with periodic coordinates
    clust = clue.clusterer(0.05, 1, 1.1)
    clust.read_data(opposite_angles, x1=(-pi, pi))
    clust.run_clue()

    # We just check that the number of clusters is two
    # If it is, the code recognizes points at the opposite extremes of the domain
    # as very far away from each other
    assert clust.clust_prop.n_clusters == 2
