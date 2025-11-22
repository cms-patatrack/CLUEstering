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

def test_clusterer_constructor():
    '''
    Test the constructor of the clusterer class
    '''
    c = clue.clusterer(21., 10., 21.)
    d = clue.clusterer(21., 10.)
    assert c._dc == d._dc
    assert c._rhoc == d._rhoc
    assert c._dm == d._dm


def test_set_params():
    c = clue.clusterer(0., 0., 0.)

    c.set_params(dc=1., rhoc=5.)
    assert c._dc == 1.
    assert c._rhoc == 5.
    assert c._dm == 1.
    assert c._seed_dc == 1.

    c.set_params(dc=1., rhoc=5., dm=2.)
    assert c._dc == 1.
    assert c._rhoc == 5.
    assert c._dm == 2.
    assert c._seed_dc == 1.

    c.set_params(dc=1., rhoc=5., dm=2., seed_dc=3)
    assert c._dc == 1.
    assert c._rhoc == 5.
    assert c._dm == 2.
    assert c._seed_dc == 3

def test_clustering_methods(dataset):
    '''
    Test the different clustering methods
    '''

    c = clue.clusterer(21., 10., 21.)
    c.read_data(dataset)
    c.run_clue()

    d = clue.clusterer(21., 10., 21.)
    d.fit(dataset)

    e = clue.clusterer(21., 10., 21.)
    cluster_ids = e.fit_predict(dataset)

    assert (c.cluster_ids == d.cluster_ids).all()
    assert (c.cluster_ids == cluster_ids).all()
    assert c.n_clusters == d.n_clusters
    assert c.n_clusters == e.n_clusters

def test_verbose_clustering(dataset):
    '''
    Test the verbose clustering
    '''

    c = clue.clusterer(21., 10., 21.)
    c.read_data(dataset)
    c.run_clue(verbose=True)

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
    labels = c.labels
    assert len(labels) == 999
    assert (cluster_ids == labels).all()
    cluster_points = c.cluster_points
    assert len(cluster_points) == nclusters
    points_per_cluster = c.points_per_cluster
    assert len(points_per_cluster) == nclusters
    for i, cluster_size in enumerate(points_per_cluster):
        assert cluster_size == len(cluster_points[i])
    output_df = c.output_df
    assert output_df.shape == (999, 1)
