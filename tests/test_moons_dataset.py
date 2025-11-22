'''
Testing the algorithm on the circle dataset, a dataset where points are distributed to form
two moon shaped clusters
'''

import os
import sys
import pandas as pd
import pytest
from check_result import check_result
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def moons():
    '''
    Returns the dataframe containing the moon dataset
    '''
    return pd.read_csv("../data/moons.csv")


def test_clustering(moons):
    '''
    Checks that the output of the clustering is the one given by the truth
    dataset.
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./moons_output.csv'):
        os.remove('./moons_output.csv')

    c = clue.clusterer(78., 80., 90., 100.)
    c.read_data(moons)
    assert c.n_dim == 2
    c.run_clue()
    c.to_csv('./', 'moons_output.csv')

    assert check_result('./moons_output.csv',
                        '../data/truth_files/moons_1000_truth.csv')


if __name__ == "__main__":
    c = clue.clusterer(78., 80, 90., 100)
    c.read_data("../data/moons.csv")
    c.run_clue()
    c.cluster_plotter()
    c.to_csv('../data/truth_files/', 'moons_1000_truth.csv')
