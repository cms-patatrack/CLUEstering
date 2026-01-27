'''
Test the algorithm for all the supported backends
'''

import os
import sys
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score
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
    return pd.read_csv("../data/sissa_1000.csv")


@pytest.fixture
def toy_det():
    '''
    Returns the dataframe containing the toy-detector dataset
    '''
    return pd.read_csv("../data/toyDetector_1000.csv")

def test_check_backends():
    '''
    Check which backends are available for the test
    '''
    if clue.is_tbb_available():
        print("TBB backend is available")
    if clue.is_openmp_available():
        print("OpenMP backend is available")
    if clue.is_cuda_available():
        print("CUDA backend is available")
    if clue.is_hip_available():
        print("HIP backend is available")


def test_blobs_clustering(blobs):
    '''
    Checks that the output of the clustering is the one given by the truth
    dataset
    '''

    for backend in clue.backends:
        c = clue.clusterer(1., 5, 2.)
        c.read_data(blobs)
        c.run_clue(backend=backend)

        mask = c.cluster_ids != -1
        assert silhouette_score(c.coords.T[mask], c.cluster_ids[mask]) > 0.8


def test_moons_clustering(moons):
    '''
    Checks that the output of the clustering is the one given by the truth
    dataset
    '''

    for backend in clue.backends:
        # Check if the output file already exists and if it does, delete it
        if os.path.isfile(f'./moons_output_{_fill_space(backend)}.csv'):
            os.remove(f'./moons_output_{_fill_space(backend)}.csv')

        c = clue.clusterer(78., 80., 90., 100.)
        c.read_data(moons)
        c.run_clue(backend=backend)
        c.to_csv('.', f'moons_output_{_fill_space(backend)}')

        assert True


def test_sissa_clustering(sissa):
    '''
    Checks that the output of the clustering is the one given by the truth
    dataset
    '''

    for backend in clue.backends:
        c = clue.clusterer(20., 10., 20.)
        c.read_data(sissa)
        c.run_clue(backend=backend)

        mask = c.cluster_ids != -1
        assert silhouette_score(c.coords.T[mask], c.cluster_ids[mask]) > 0.5


def test_toydet_clustering(toy_det):
    '''
    Checks that the output of the clustering is the one given by the truth
    dataset
    '''

    for backend in clue.backends:
        c = clue.clusterer(4., 2.5, 4.)
        c.read_data(toy_det)
        c.run_clue(backend=backend)

        mask = c.cluster_ids != -1
        assert silhouette_score(c.coords.T[mask], c.cluster_ids[mask]) > 0.8


if __name__ == "__main__":
    c = clue.clusterer(1., 5, 2.)
    c.read_data("../data/blob.csv")
    c.run_clue()
    c.cluster_plotter()

    c = clue.clusterer(78., 80., 90., 100.)
    c.read_data("../data/moons.csv")
    c.run_clue()
    c.cluster_plotter()

    c = clue.clusterer(20., 10., 20.)
    c.read_data("../data/sissa_1000.csv")
    c.run_clue()
    c.cluster_plotter()

    c = clue.clusterer(4., 2.5, 4.)
    c.read_data("../data/toyDetector_1000.csv")
    c.run_clue()
    c.cluster_plotter()
