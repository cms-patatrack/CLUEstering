'''
Testing the algorithm on the circle dataset, a dataset where points are
distributed to simulate the hits of a small set of particles in a detector.
'''

import os
import sys
import pandas as pd
import pytest
from check_result import check_result
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


@pytest.fixture
def toy_det():
    '''
    Returns the dataframe containing the toy-detector dataset
    '''
    return pd.read_csv("./test_datasets/toyDetector.csv")


def test_clustering(toy_det):
    '''
    Checks that the output of the clustering is the one given by the
    truth dataset.
    '''

    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./toy_det_output.csv'):
        os.remove('./toy_det_output.csv')

    c = clue.clusterer(4.5, 2.5, 1.)
    c.read_data(toy_det)
    c.run_clue()
    c.to_csv('./', 'toy_det_output.csv')

    assert check_result('./toy_det_output.csv',
                        './test_datasets/truth_files/toy_det_1000_truth.csv')


if __name__ == "__main__":
    c = clue.clusterer(4.5, 2.5, 1.)
    c.read_data("./test_datasets/toyDetector.csv")
    c.run_clue()
    c.cluster_plotter()
