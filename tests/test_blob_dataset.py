'''
Testing the algorithm on the blob dataset, a dataset where points are distributed to form
round clusters
'''

import os
import sys
import pandas as pd
import pytest
sys.path.insert(1, '.')
from check_result import check_result
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue

@pytest.fixture
def blobs():
    '''
    Returns the dataframe containing the blob dataset
    '''
    return pd.read_csv("./test_datasets/blob.csv")


def test_blobs_clustering(blobs):
    '''
    Checks that the output of the clustering is the one given by the truth dataset
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./blobs_output.csv'):
        os.remove('./blobs_output.csv')

    c = clue.clusterer(0.8, 5, 0.8)
    c.read_data(blobs)
    c.run_clue()
    c.to_csv('./', 'blobs_output.csv')

    check_result('./blobs_output.csv',
                 './test_datasets/truth_files/blobs_truth.csv')

if __name__ == "__main__":
    c = clue.clusterer(0.8, 5, 0.8)
    c.read_data("./test_datasets/blob.csv")
    c.run_clue()
    c.cluster_plotter()
