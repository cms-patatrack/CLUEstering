'''
Test clustering on a dataset with wrapped coordinates.
'''

import os
import sys
import pandas as pd
import pytest
from check_result import check_result
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue

@pytest.fixture
def dataset():
    '''
    Returns the dataframe containing the sissa dataset 1000 points
    '''
    return pd.read_csv("../data/opposite_angles.csv")

def test_wrapped_coordinates(dataset):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset.
    '''

    c = clue.clusterer(0.2, 5., 0.2)
    c.read_data(dataset, wrapped_coords=[0, 1])
    assert c.n_dim == 2
    c.run_clue()
    assert c.n_clusters == 1

    d = clue.clusterer(0.2, 5., 0.2)
    d.read_data(dataset)
    d.set_wrapped([0, 1])
    assert d.n_dim == 2
    d.run_clue()
    assert d.n_clusters == 1

def test_without_wrapped_coordinates(dataset):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset.
    '''

    c = clue.clusterer(0.2, 5., 0.2)
    c.read_data(dataset)
    assert c.n_dim == 2
    c.run_clue()
    assert c.n_clusters == 2
