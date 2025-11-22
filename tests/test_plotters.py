'''
Test the clusterer's plotting methods
'''

import matplotlib
import os
import sys
import pandas as pd
import pytest
from check_result import check_result
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue
matplotlib.use("Agg")


@pytest.fixture
def dataset1d():
    '''
    Returns a 1d dataset for testing
    '''

    data = {'x': [1, 2, 1.5, 8, 9, 8.5, 50, 51, 49.5], 'weight': [1]*9}
    return pd.DataFrame(data)


@pytest.fixture
def dataset2d():
    '''
    Returns a 2d dataset for testing
    '''

    data = {'x': [1, 2, 1.5, 8, 9, 8.5, 50, 51, 49.5],
            'y': [1, 2, 1.5, 8, 9, 8.5, 50, 51, 49.5],
            'weight': [1]*9}
    return pd.DataFrame(data)


@pytest.fixture
def dataset3d():
    '''
    Returns a generic dataset for testing
    '''

    data = {'x': [1, 2, 1.5, 8, 9, 8.5, 50, 51, 49.5],
            'y': [1, 2, 1.5, 8, 9, 8.5, 50, 51, 49.5],
            'z': [1, 2, 1.5, 8, 9, 8.5, 50, 51, 49.5],
            'weight': [1]*9}
    return pd.DataFrame(data)


def test_input_plotter(dataset1d, dataset2d, dataset3d):
    '''
    Tests the input plotter method of the clusterer
    '''

    c = clue.clusterer(3., 2., 3.)

    xticks = [0, 10, 20, 30, 40, 50, 60]
    yticks = [0, 10, 20, 30, 40, 50, 60]
    zticks = [0, 10, 20, 30, 40, 50, 60]

    filename = 'dataset1d.png'
    c.read_data(dataset1d)
    assert c.n_dim == 1
    c.input_plotter()
    c.input_plotter(grid=True, xticks=xticks)
    c.input_plotter(filename)
    assert os.path.isfile(filename)

    filename = 'dataset2d.png'
    c.read_data(dataset2d)
    assert c.n_dim == 2
    c.input_plotter()
    c.input_plotter(grid=True, xticks=xticks, yticks=yticks)
    c.input_plotter(filename)
    assert os.path.isfile(filename)

    filename = 'dataset3d.png'
    c.read_data(dataset3d)
    assert c.n_dim == 3
    c.input_plotter()
    c.input_plotter(grid=True, xticks=xticks, yticks=yticks, zticks=zticks)
    c.input_plotter(filename)
    assert os.path.isfile(filename)

def test_output_plotter(dataset1d, dataset2d, dataset3d):
    '''
    Tests the output plotter method of the clusterer
    '''

    c = clue.clusterer(3., 2., 3.)

    xticks = [0, 10, 20, 30, 40, 50, 60]
    yticks = [0, 10, 20, 30, 40, 50, 60]
    zticks = [0, 10, 20, 30, 40, 50, 60]

    filename = 'output1d.png'
    c.read_data(dataset1d)
    assert c.n_dim == 1
    c.run_clue()
    c.cluster_plotter()
    c.cluster_plotter(grid=True, xticks=xticks)
    c.cluster_plotter(filename)
    assert os.path.isfile(filename)

    filename = 'output2d.png'
    c.read_data(dataset2d)
    assert c.n_dim == 2
    c.run_clue()
    c.cluster_plotter()
    c.cluster_plotter(grid=True, xticks=xticks, yticks=yticks)
    c.cluster_plotter(filename)
    assert os.path.isfile(filename)

    filename = 'output3d.png'
    c.read_data(dataset3d)
    assert c.n_dim == 3
    c.run_clue()
    c.cluster_plotter()
    c.cluster_plotter(grid=True, xticks=xticks, yticks=yticks, zticks=zticks)
    c.cluster_plotter(filename)
    assert os.path.isfile(filename)
