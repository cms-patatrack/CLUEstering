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


@pytest.fixture
def square():
    '''
    Retuns a dataframe where the points are distributed in a square
    '''
    return pd.read_csv("./test_datasets/square.csv")


@pytest.fixture
def box():
    '''
    Retuns a dataframe where the points are distributed in a 3D box
    '''
    return pd.read_csv("./test_datasets/box.csv")


def test_one_out_of_two(moons):
    '''
    Test the dimension reduction from 2D to 1D
    '''

    c = clue.clusterer(.4, 2., 1.6)
    c.read_data(moons)
    # check the initial number of dimensions
    assert len(c.clust_data.coords[0]) == 2
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
    '''
    Test the dimension reduction from 3D to 1D
    '''

    c = clue.clusterer(.4, 2., 1.6)
    c.read_data(blobs)
    # check the initial number of dimensions
    assert len(c.clust_data.coords[0]) == 3
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
    '''
    Test the dimension reduction from 3D to 2D
    '''

    c = clue.clusterer(.4, 2., 1.6)
    c.read_data(blobs)
    # check the initial number of dimensions
    assert len(c.clust_data.coords[0]) == 3
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


def test_square_box(square, box):
    '''
    Compare the clustering of a 2D square with that of a 3D box 
    clustered using only two dimensions
    '''
    c1 = clue.clusterer(1., 2., 1.6)
    c1.read_data(square)
    c1.run_clue()

    c2 = clue.clusterer(1., 2., 1.6)
    c2.read_data(box)
    c2.run_clue(dimensions=[0, 1])

    # check that the result of clustering the 3D dataset using only
    # two dimensions is the same as clustering the 2D dataset
    assert c1.clust_prop == c2.clust_prop
