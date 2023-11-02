'''
Testing the algorithm on the circle dataset, a dataset where points are distributed to form
two moon shaped clusters
'''

from filecmp import cmp
import CLUEstering as clue
import os
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')


@pytest.fixture
def moons():
    '''
    Returns the dataframe containing the moon dataset
    '''
    return pd.read_csv("./test_datasets/moons.csv")


def test_circles_clustering(moons):
    '''
    Checks that the output of the clustering is the one given by the truth dataset
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./moons_output.csv'):
        os.remove('./moons_output.csv')

    c = clue.clusterer(0.5, 5, 1.)
    c.read_data(moons)
    c.run_clue()
    c.to_csv('./', 'moons_output.csv')

    assert cmp('./moons_output.csv',
               './test_datasets/truth_files/moons_1000_truth.csv')
