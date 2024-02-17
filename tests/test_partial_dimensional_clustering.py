'''
'''

import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue

@pytest.fixture
def moons():
    '''
    Returns the dataframe containing the moon dataset
    '''
    return pd.read_csv("./test_datasets/moons.csv")

@pytest.fixture
def blobs():
    '''
    Returns the dataframe containing the blob dataset
    '''
    return pd.read_csv("./test_datasets/blob.csv")

def test_one_out_of_two(moons):
    c = clue.clusterer(.4, 2., 1.6)
    c.read_data(moons)
    coords_x0 = c._partial_dimension_dataset([0])
    coords_x1 = c._partial_dimension_dataset([1])
    
    for i in range(c.clust_data.n_points):
        # check that the coordinates are the same
        assert coords_x0[i] == c.clust_data.coords[i][0]
        assert coords_x1[i] == c.clust_data.coords[i][1]
        # check that the number of dimensions is correct
        assert len(coords_x0[i]) == 1
        assert len(coords_x1[i]) == 1
    # check that the number of points is correct
    assert len(coords_x0) == c.clust_data.n_points
    assert len(coords_x1) == c.clust_data.n_points

def test_one_out_of_three(blobs):
    c = clue.clusterer(.4, 2., 1.6)
    c.read_data(blobs)
    coords_x0 = c._partial_dimension_dataset([0])
    coords_x1 = c._partial_dimension_dataset([1])
    coords_x2 = c._partial_dimension_dataset([2])

    for i in range(c.clust_data.n_points):
        # check that the coordinates are the same
        assert coords_x0[i] == c.clust_data.coords[i][0]
        assert coords_x1[i] == c.clust_data.coords[i][1]
        assert coords_x2[i] == c.clust_data.coords[i][2]
        # check that the number of dimensions is correct
        assert len(coords_x0[i]) == 1
        assert len(coords_x1[i]) == 1
        assert len(coords_x2[i]) == 1
    # check that the number of points is correct
    assert len(coords_x0) == c.clust_data.n_points
    assert len(coords_x1) == c.clust_data.n_points
    assert len(coords_x2) == c.clust_data.n_points

def test_two_out_of_three(blobs):
    c = clue.clusterer(.4, 2., 1.6)
    c.read_data(blobs)
    print(c.clust_data.coords)
    coords_x0x1 = c._partial_dimension_dataset([0, 1])
    coords_x0x2 = c._partial_dimension_dataset([0, 2])
    coords_x1x2 = c._partial_dimension_dataset([1, 2])

    for i in range(c.clust_data.n_points):
        # check that the coordinates are the same
        assert coords_x0x1[i][0] == c.clust_data.coords[i][0]
        assert coords_x0x1[i][1] == c.clust_data.coords[i][1]
        assert coords_x0x2[i][0] == c.clust_data.coords[i][0]
        assert coords_x0x2[i][1] == c.clust_data.coords[i][2]
        assert coords_x1x2[i][0] == c.clust_data.coords[i][1]
        assert coords_x1x2[i][1] == c.clust_data.coords[i][2]
        # check that the number of dimensions is correct
        assert len(coords_x0x1[i]) == 2
        assert len(coords_x0x2[i]) == 2
        assert len(coords_x0x2[i]) == 2
    # check that the number of points is correct
    assert len(coords_x0x1) == c.clust_data.n_points
    assert len(coords_x0x2) == c.clust_data.n_points
    assert len(coords_x1x2) == c.clust_data.n_points
