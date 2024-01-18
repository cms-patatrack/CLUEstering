'''
Testing the algorithm on the circle dataset, a dataset where points are distributed to form
many small clusters
'''

from filecmp import cmp
import os
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def sissa():
    '''
    Returns the dataframe containing the sissa dataset
    '''
    return pd.read_csv("./test_datasets/sissa.csv")


def test_circles_clustering(sissa):
    '''
    Checks that the output of the clustering is the one given by the truth dataset
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./sissa_output.csv'):
        os.remove('./sissa_output.csv')

    c = clue.clusterer(0.4, 5, 1.)
    c.read_data(sissa)
    c.run_clue()
    c.to_csv('./', 'sissa_output.csv')

    assert cmp('./sissa_output.csv',
               './test_datasets/truth_files/sissa_1000_truth.csv')
