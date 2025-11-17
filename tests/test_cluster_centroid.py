'''
Test clusterer methods for computing cluster centroids
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
    return pd.read_csv("../data/sissa_1000.csv")

def test_cluster_centroids(dataset):
    '''
    Test the methods for computing cluster centroids
    '''
    c = clue.clusterer(21., 10., 21.)
    c.read_data(dataset)
    c.run_clue()

    centroids = c.cluster_centroids()
    assert centroids.shape == (c.n_clusters, c.n_dim)
    single_centroids = []
    for cluster in range(c.n_clusters):
        print(cluster)
        centroid = c.cluster_centroid(cluster)
        single_centroids.append(centroid)

    assert (single_centroids == centroids).all()
