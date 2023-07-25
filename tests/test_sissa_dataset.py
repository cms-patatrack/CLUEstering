import numpy as np
import os
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue
from filecmp import cmp

@pytest.fixture
def sissa():
    return pd.read_csv("./test_datasets/sissa_1000.csv")

def test_circles_clustering(sissa):
    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./sissa_output.csv'):
        os.remove('./sissa_output.csv')

    c = clue.clusterer(0.4,5,1.)
    c.read_data(sissa)
    c.run_clue()
    c.to_csv('./','sissa_output.csv')

    assert cmp('./sissa_output.csv', './test_datasets/sissa_1000_truth.csv')
