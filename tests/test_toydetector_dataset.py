import numpy as np
import os
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue
from filecmp import cmp

@pytest.fixture
def toy_det():
    return pd.read_csv("./test_datasets/toyDetector_1000.csv")

def test_circles_clustering(toy_det):
    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./moons_output.csv'):
        os.remove('./moons_output.csv')

    c = clue.clusterer(0.06,5,1.)
    c.read_data(toy_det)
    c.run_clue()
    c.to_csv('./','toy_det_output.csv')

    assert cmp('./toy_det_output.csv', './test_datasets/toy_det_1000_truth.csv')

if __name__ == "__main__":
    c = clue.clusterer(0.06,5,1.)
    c.read_data('./test_datasets/toyDetector_1000.csv')
    c.run_clue()
    c.cluster_plotter()
