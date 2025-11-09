'''
Testing the algorithms on a dataset with three anisotropically distributed clusters.
'''

import os
import sys
import pandas as pd
import pytest
from check_result import check_result
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue

@pytest.fixture
def aniso():
    '''
    Returns the aniso dataframe
    '''
    return pd.read_csv("../data/aniso_1000.csv")

def test_clustering(aniso):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset.
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./aniso_output.csv'):
        os.remove('./aniso_output.csv')

    c = clue.clusterer(25., 5., 23.)
    c.read_data(aniso)
    assert c.n_dim == 2
    c.run_clue()
    c.to_csv('./', 'aniso_output.csv')

    assert check_result('./aniso_output.csv',
                        '../data/truth_files/aniso_1000_truth.csv')

if __name__ == "__main__":
    c = clue.clusterer(25., 5., 23.)
    c.read_data('../data/aniso_1000.csv')
    c.run_clue()
    c.cluster_plotter()
