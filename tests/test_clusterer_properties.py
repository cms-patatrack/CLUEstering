'''
Testing the properties of the clusterer class
'''

import os
import sys
import numpy as np
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

def test_clusterer_properties(dataset):
    '''
    Test the properties of the clusterer class
    '''
    c = clue.clusterer(21., 10., 21.)
    c.read_data(dataset)
    dataset_array = dataset.to_numpy()

    coords = c.coords
    assert coords.shape == (2, 999)
    weights = c.weight
    assert weights.shape == (999,)
    assert (weights == 1.).all()
    assert (weights == dataset_array.T[2]).all()
    ndim = c.n_dim
    assert ndim == 2
    npoints = c.n_points
    assert npoints == 999

    c.run_clue()

    nclusters = c.n_clusters
    assert nclusters > 0
    cluster_ids = c.cluster_ids
    assert np.max(cluster_ids) + 1 == nclusters
    clusters = c.clusters
    assert len(clusters) == nclusters
    labels = c.labels
    assert len(labels) == 999
    assert (cluster_ids == labels).all()
    cluster_points = c.cluster_points
    assert len(cluster_points) == nclusters
    output_df = c.output_df
    assert output_df.shape == (999, 2)
