import numpy as np
import os
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue
from filecmp import cmp

@pytest.fixture
def moons():
    return pd.read_csv("./test_datasets/moons_1000.csv")

def test_circles_clustering(moons):
    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./moons_output.csv'):
        os.remove('./moons_output.csv')

    c = clue.clusterer(0.5,5,1.)
    c.read_data(moons)
    c.run_clue()
    c.to_csv('./','moons_output.csv')

    assert cmp('./moons_output.csv', './test_datasets/moons_1000_truth.csv')

if __name__ == "__main__":
    c = clue.clusterer(0.5,5,1.)
    c.read_data('./test_datasets/moons_1000.csv')
    c.run_clue()
    c.cluster_plotter()
