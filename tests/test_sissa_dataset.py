'''
Testing the algorithm on the sissa datasets
'''

import os
import sys
import pandas as pd
import pytest
from check_result import check_result
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def sissa_1000():
    '''
    Returns the dataframe containing the sissa dataset 1000 points
    '''
    return pd.read_csv("../data/sissa_1000.csv")


@pytest.fixture
def sissa_4000():
    '''
    Returns the dataframe containing the sissa dataset with 4000 points
    '''
    return pd.read_csv("../data/sissa_4000.csv")


def test_sissa_1000(sissa_1000):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset.
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./sissa_1000_output.csv'):
        os.remove('./sissa_1000_output.csv')

    c = clue.clusterer(21., 10., 21.)
    c.read_data(sissa_1000)
    assert c.n_dim == 2
    c.run_clue()
    c.to_csv('./', 'sissa_1000_output.csv')

    assert check_result('./sissa_1000_output.csv',
                        '../data/truth_files/sissa_1000_truth.csv')


def test_sissa_4000(sissa_4000):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset.
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./sissa_4000_output.csv'):
        os.remove('./sissa_4000_output.csv')

    c = clue.clusterer(20., 10., 20.)
    c.read_data(sissa_4000)
    assert c.n_dim == 2
    c.run_clue()
    c.to_csv('./', 'sissa_4000_output.csv')

    assert check_result('./sissa_4000_output.csv',
                        '../data/truth_files/sissa_4000_truth.csv')



if __name__ == "__main__":
    c = clue.clusterer(21., 10., 21.)
    c.read_data("../data/sissa_1000.csv")
    c.run_clue()
    c.cluster_plotter()

    c = clue.clusterer(20., 10., 20.)
    c.read_data("../data/sissa_4000.csv")
    c.run_clue()
    c.cluster_plotter()
