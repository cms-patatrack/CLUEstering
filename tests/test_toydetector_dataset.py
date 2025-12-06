'''
Testing the algorithm on toy detector datasets, datasets where points are
distributed to simulate the hits of a small set of particles in a detector.
'''

import os
import sys
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def toy_det_1000():
    '''
    Returns the dataframe containing the toy-detector dataset with 1000 points
    '''
    return pd.read_csv("../data/toyDetector_1000.csv")


@pytest.fixture
def toy_det_5000():
    '''
    Returns the dataframe containing the toy-detector dataset with 1000 points
    '''
    return pd.read_csv("../data/toyDetector_5000.csv")


@pytest.fixture
def toy_det_10000():
    '''
    Returns the dataframe containing the toy-detector dataset with 1000 points
    '''
    return pd.read_csv("../data/toyDetector_10000.csv")


def test_toy_det_1000(toy_det_1000):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset.
    '''

    c = clue.clusterer(4., 2.5, 4.)
    c.read_data(toy_det_1000)
    assert c.n_dim == 2
    c.run_clue()

    mask = c.cluster_ids != -1
    assert silhouette_score(c.coords.T[mask], c.cluster_ids[mask]) > 0.8


# TODO: Uncomment after rewriting `check_result`
# def test_toy_det_5000(toy_det_5000):
#     '''
#     Checks that the output of the clustering is the one given by the
#     truth dataset.
#     '''

#     # Check if the output file already exists and if it does, delete it
#     if os.path.isfile('./toy_det_5000_output.csv'):
#         os.remove('./toy_det_5000_output.csv')

#     c = clue.clusterer(2.5, 2., 7.5)
#     c.read_data(toy_det_5000)
#     assert c.n_dim == 2
#     c.run_clue()
#     c.to_csv('./', 'toy_det_5000_output.csv')

#     assert check_result('./toy_det_5000_output.csv',
#                         '../data/truth_files/toy_det_5000_truth.csv')


# TODO: Uncomment after rewriting `check_result`
# def test_toy_det_10000(toy_det_10000):
#     '''
#     Checks that the output of the clustering is the one given by the
#     truth dataset.
#     '''

#     # Check if the output file already exists and if it does, delete it
#     if os.path.isfile('./toy_det_10000_output.csv'):
#         os.remove('./toy_det_10000_output.csv')

#     c = clue.clusterer(2.5, 2., 8.5)
#     c.read_data(toy_det_10000)
#     assert c.n_dim == 2
#     c.run_clue()
#     c.to_csv('./', 'toy_det_10000_output.csv')

#     assert check_result('./toy_det_10000_output.csv',
#                         '../data/truth_files/toy_det_10000_truth.csv')


if __name__ == "__main__":
    c = clue.clusterer(4., 2.5, 4.)
    c.read_data("../data/toyDetector_1000.csv")
    c.run_clue()
    c.cluster_plotter()
