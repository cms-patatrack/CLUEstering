'''
Testing the distance metrics
'''

import os
import sys
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def sissa_1000():
    '''
    Returns the dataframe containing the sissa dataset 1000 points
    '''
    return pd.read_csv("../data/sissa_1000.csv")


def test_euclidean(sissa_1000):
    '''
    Tests the euclidean metric
    '''

    c = clue.clusterer(20., 10., 20.)
    c.read_data(sissa_1000)
    assert c.n_dim == 2
    c.choose_metric('euclidean')
    c.run_clue()

    mask = c.cluster_ids != -1
    assert silhouette_score(c.coords.T[mask], c.cluster_ids[mask]) > 0.5


def test_weighted_euclidean(sissa_1000):
    '''
    Tests the weighted euclidean metric
    '''

    c = clue.clusterer(20., 10., 20.)
    c.read_data(sissa_1000)
    assert c.n_dim == 2
    c.choose_metric('weighted_euclidean', [0.5, 0.5])
    c.run_clue()

    mask = c.cluster_ids != -1
    assert silhouette_score(c.coords.T[mask], c.cluster_ids[mask]) > 0.5


def test_manhattan(sissa_1000):
    '''
    Tests the manhattan metric
    '''

    c = clue.clusterer(20., 10., 20.)
    c.read_data(sissa_1000)
    assert c.n_dim == 2
    c.choose_metric('manhattan')
    c.run_clue()

    mask = c.cluster_ids != -1
    assert silhouette_score(c.coords.T[mask], c.cluster_ids[mask]) > 0.5


def test_chebyshev(sissa_1000):
    '''
    Tests the chebyshev metric
    '''

    c = clue.clusterer(20., 10., 20.)
    c.read_data(sissa_1000)
    assert c.n_dim == 2
    c.choose_metric('chebyshev')
    c.run_clue()

    mask = c.cluster_ids != -1
    assert silhouette_score(c.coords.T[mask], c.cluster_ids[mask]) > 0.5

def test_weighted_chebyshev(sissa_1000):
    '''
    Tests the weighted chebyshev metric
    '''

    c = clue.clusterer(20., 10., 20.)
    c.read_data(sissa_1000)
    assert c.n_dim == 2
    c.choose_metric('weighted_chebyshev', [0.5, 0.5])
    c.run_clue()

    mask = c.cluster_ids != -1
    assert silhouette_score(c.coords.T[mask], c.cluster_ids[mask]) > 0.5
