'''
'''

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
    return pd.read_csv("../data/moons.csv")


@pytest.fixture
def blobs():
    '''
    Returns the dataframe containing the blob dataset
    '''
    return pd.read_csv("../data/blob.csv")


@pytest.fixture
def square():
    '''
    Retuns a dataframe where the points are distributed in a square
    '''
    return pd.read_csv("../data/square.csv")


@pytest.fixture
def box():
    '''
    Retuns a dataframe where the points are distributed in a 3D box
    '''
    return pd.read_csv("../data/box.csv")


def test_one_out_of_two(moons):
    '''
    Test the dimension reduction from 2D to 1D
    '''

    c = clue.clusterer(0.4, 2., 0.4)
    c.read_data(moons)
    assert c.n_dim == 2
    # check the initial number of dimensions
    assert len(c.clust_data.coords[0]) == 999
    coordsoa_x0 = c._partial_dimension_dataset([0])
    coordsoa_x1 = c._partial_dimension_dataset([1])

    assert (coordsoa_x0[0] == c.clust_data.coords[0]).all()
    assert (coordsoa_x1[0] == c.clust_data.coords[1]).all()
    assert len(coordsoa_x0) == 2
    assert len(coordsoa_x1) == 2
    # check that the number of points is correct
    assert len(coordsoa_x0[0]) == c.clust_data.n_points
    assert len(coordsoa_x1[0]) == c.clust_data.n_points


def test_one_out_of_three(blobs):
    '''
    Test the dimension reduction from 3D to 1D
    '''

    c = clue.clusterer(0.4, 2., 0.4)
    c.read_data(blobs)
    assert c.n_dim == 3
    # check the initial number of dimensions
    assert len(c.clust_data.coords[0]) == 10000
    coordsoa_x0 = c._partial_dimension_dataset([0])
    coordsoa_x1 = c._partial_dimension_dataset([1])
    coordsoa_x2 = c._partial_dimension_dataset([2])

    assert (coordsoa_x0[0] == c.clust_data.coords[0]).all()
    assert (coordsoa_x1[0] == c.clust_data.coords[1]).all()
    assert (coordsoa_x2[0] == c.clust_data.coords[2]).all()
    assert len(coordsoa_x0) == 2
    assert len(coordsoa_x1) == 2
    assert len(coordsoa_x2) == 2
    # check that the number of points is correct
    assert len(coordsoa_x0[0]) == c.clust_data.n_points
    assert len(coordsoa_x1[0]) == c.clust_data.n_points
    assert len(coordsoa_x2[0]) == c.clust_data.n_points


def test_two_out_of_three(blobs):
    '''
    Test the dimension reduction from 3D to 2D
    '''

    c = clue.clusterer(0.4, 2., 0.4)
    c.read_data(blobs)
    assert c.n_dim == 3
    # check the initial number of dimensions
    assert len(c.clust_data.coords[0]) == 10000
    coordsoa_x0x1 = c._partial_dimension_dataset([0, 1])
    coordsoa_x0x2 = c._partial_dimension_dataset([0, 2])
    coordsoa_x1x2 = c._partial_dimension_dataset([1, 2])

    assert (coordsoa_x0x1[0] == c.clust_data.coords[0]).all()
    assert (coordsoa_x0x1[1] == c.clust_data.coords[1]).all()
    assert (coordsoa_x0x2[0] == c.clust_data.coords[0]).all()
    assert (coordsoa_x0x2[1] == c.clust_data.coords[2]).all()
    assert (coordsoa_x1x2[0] == c.clust_data.coords[1]).all()
    assert (coordsoa_x1x2[1] == c.clust_data.coords[2]).all()
    assert len(coordsoa_x0x1) == 3
    assert len(coordsoa_x0x2) == 3
    assert len(coordsoa_x0x2) == 3

    assert len(coordsoa_x0x1[0]) == c.clust_data.n_points
    assert len(coordsoa_x0x2[0]) == c.clust_data.n_points
    assert len(coordsoa_x1x2[0]) == c.clust_data.n_points


def test_square_box(square, box):
    '''
    Compare the clustering of a 2D square with that of a 3D box
    clustered using only two dimensions
    '''
    c1 = clue.clusterer(1., 2., 1.6)
    c1.read_data(square)
    assert c1.n_dim == 2
    c1.run_clue()

    c2 = clue.clusterer(1., 2., 1.6)
    c2.read_data(box)
    assert c2.n_dim == 3
    c2.run_clue(dimensions=[0, 1])

    # check that the result of clustering the 3D dataset using only
    # two dimensions is the same as clustering the 2D dataset
    assert c1.clust_prop == c2.clust_prop

if __name__ == "__main__":
    c = clue.clusterer(1., 2., 1.6)
    c.read_data(pd.read_csv("../data/box.csv"))
    c.run_clue(dimensions=[0, 1])
    c.cluster_plotter()
