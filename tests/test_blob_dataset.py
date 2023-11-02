'''
Testing the algorithm on the blob dataset, a dataset where points are distributed to form
round clusters
'''

from filecmp import cmp
import CLUEstering as clue
import numpy as np
import os
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')


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

    c = clue.clusterer(0.8, 5, 1.5)
    c.read_data(blobs)
    c.run_clue()
    c.to_csv('./', 'blobs_output.csv')

    assert cmp('./blobs_output.csv',
               './test_datasets/truth_files/blobs_truth.csv')
