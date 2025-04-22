'''
Test the import of a clusterer from csv file
'''

import os
import sys
import pandas as pd
import pytest
from check_result import check_result
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue

@pytest.fixture
def sissa():
    '''
    Returns the dataframe containing the sissa dataset
    '''
    return pd.read_csv("../data/sissa.csv")

def test_clusterer_import(sissa):
    '''
    Try importing a clusterer from csv file and check that it's equal to the original clusterer
    '''
    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./test_sissa_import.csv'):
        os.remove('./test_sissa_import.csv')

    c = clue.clusterer(20., 10., 20.)
    c.read_data(sissa)
    c.run_clue()
    c.to_csv('./', 'test_sissa_import.csv')

    d = clue.clusterer(20., 10., 20.)
    d.import_clusterer('./', 'test_sissa_import.csv')

    assert c.clust_prop == d.clust_prop

if __name__ == "__main__":
    c = clue.clusterer(20., 10., 20.)
    c.read_data("../data/sissa.csv")
    c.run_clue()
    c.cluster_plotter()
    c.to_csv('./', 'test_sissa_import.csv')

    d = clue.clusterer(20., 10., 20.)
    d.import_clusterer('./', 'test_sissa_import.csv')
    d.cluster_plotter()
