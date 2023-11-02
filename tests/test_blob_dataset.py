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
    return pd.read_csv("./test_datasets/blob.csv")


def test_blobs_clustering(blobs):
    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./blobs_output.csv'):
        os.remove('./blobs_output.csv')

    c = clue.clusterer(0.8, 5, 1.5)
    c.read_data(blobs)
    c.run_clue()
    c.to_csv('./', 'blobs_output.csv')

    assert cmp('./blobs_output.csv',
               './test_datasets/truth_files/blobs_truth.csv')
