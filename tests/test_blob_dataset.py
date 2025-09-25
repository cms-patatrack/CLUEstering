'''
Testing the algorithm on the blob dataset, a dataset where points are
distributed to form round clusters
'''

import os
import sys
import pandas as pd
import pytest
from check_result import check_result
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def blobs():
    '''
    Returns the dataframe containing the blob dataset
    '''
    return pd.read_csv("../data/blob.csv")


def test_clustering(blobs):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./blobs_output.csv'):
        os.remove('./blobs_output.csv')

    c = clue.clusterer(1., 5, 2.)
    c.read_data(blobs)
    assert c.n_dim == 3
    c.run_clue()
    c.to_csv('./', 'blobs_output.csv')

    assert check_result('./blobs_output.csv',
                        '../data/truth_files/blobs_truth.csv')


if __name__ == "__main__":
    c = clue.clusterer(1., 5, 2.)
    c.read_data("../data/blob.csv")
    c.run_clue()
    c.cluster_plotter()
