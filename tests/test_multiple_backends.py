'''
Test the algorithm for all the supported backends
'''

import os
import sys
import pandas as pd
import pytest
from check_result import check_result
sys.path.insert(1, '.')
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


def _fill_space(string: str) -> str:
    '''
    Substitutes the spaces in a string with underscores
    '''

    return string.replace(' ', '_')


@pytest.fixture
def blobs():
    '''
    Returns the dataframe containing the blob dataset
    '''
    return pd.read_csv("../data/blob.csv")


@pytest.fixture
def moons():
    '''
    Returns the dataframe containing the moon dataset
    '''
    return pd.read_csv("../data/moons.csv")


@pytest.fixture
def sissa():
    '''
    Returns the dataframe containing the sissa dataset
    '''
    return pd.read_csv("../data/sissa.csv")


@pytest.fixture
def toy_det():
    '''
    Returns the dataframe containing the toy-detector dataset
    '''
    return pd.read_csv("../data/toyDetector.csv")


def test_blobs_clustering(blobs):
    '''
    Checks that the output of the clustering is the one given by the truth
    dataset
    '''

    for backend in clue.backends:
        # Check if the output file already exists and if it does, delete it
        if os.path.isfile(f'./blobs_output_{_fill_space(backend)}.csv'):
            os.remove(f'./blobs_output_{_fill_space(backend)}.csv')

        c = clue.clusterer(1., 5, 2.)
        c.read_data(blobs)
        c.run_clue(backend=backend)
        c.to_csv('./', f'blobs_output_{_fill_space(backend)}.csv')

        assert check_result(f'./blobs_output_{_fill_space(backend)}.csv',
                            '../data/truth_files/blobs_truth.csv')


def test_moons_clustering(moons):
    '''
    Checks that the output of the clustering is the one given by the truth
    dataset
    '''

    for backend in clue.backends:
        # Check if the output file already exists and if it does, delete it
        if os.path.isfile(f'./moons_output_{_fill_space(backend)}.csv'):
            os.remove(f'./moons_output_{_fill_space(backend)}.csv')

        c = clue.clusterer(50., 5., 120.)
        c.read_data(moons)
        c.run_clue(backend=backend)
        c.to_csv('./', f'moons_output_{_fill_space(backend)}.csv')

        assert check_result(f'./moons_output_{_fill_space(backend)}.csv',
                            '../data/truth_files/moons_1000_truth.csv')


def test_sissa_clustering(sissa):
    '''
    Checks that the output of the clustering is the one given by the truth
    dataset
    '''

    for backend in clue.backends:
        # Check if the output file already exists and if it does, delete it
        if os.path.isfile(f'./sissa_output_{_fill_space(backend)}.csv'):
            os.remove(f'./sissa_output_{_fill_space(backend)}.csv')

        c = clue.clusterer(21., 10., 21.)
        c.read_data(sissa)
        c.run_clue(backend=backend)
        c.to_csv('./', f'sissa_output_{_fill_space(backend)}.csv')

        assert check_result(f'./sissa_output_{_fill_space(backend)}.csv',
                            '../data/truth_files/sissa_1000_truth.csv')


def test_toydet_clustering(toy_det):
    '''
    Checks that the output of the clustering is the one given by the truth
    dataset
    '''

    for backend in clue.backends:
        # Check if the output file already exists and if it does, delete it
        if os.path.isfile(f'./toy_det_output_{_fill_space(backend)}.csv'):
            os.remove(f'./toy_det_output_{_fill_space(backend)}.csv')

        c = clue.clusterer(4., 2.5, 4.)
        c.read_data(toy_det)
        c.run_clue(backend=backend)
        c.to_csv('./', f'toy_det_output_{_fill_space(backend)}.csv')

        assert check_result(f'./toy_det_output_{_fill_space(backend)}.csv',
                            '../data/truth_files/toy_det_1000_truth.csv')


if __name__ == "__main__":
    c = clue.clusterer(1., 5, 2.)
    c.read_data("../data/blob.csv")
    c.run_clue()
    c.cluster_plotter()

    c = clue.clusterer(50., 5., 120.)
    c.read_data("../data/moons.csv")
    c.run_clue()
    c.cluster_plotter()

    c = clue.clusterer(21., 10., 21.)
    c.read_data("../data/sissa.csv")
    c.run_clue()
    c.cluster_plotter()

    c = clue.clusterer(4., 2.5, 4.)
    c.read_data("../data/toyDetector.csv")
    c.run_clue()
    c.cluster_plotter()
