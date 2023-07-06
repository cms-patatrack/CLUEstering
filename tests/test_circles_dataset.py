import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue
from filecmp import cmp

@pytest.fixture
def circles():
    return pd.read_csv("./test_datasets/circles_1000.csv")

def test_circles_clustering(circles):
    c = clue.clusterer(0.9,5,1.5)
    c.read_data(circles)
    c.change_coordinates(x0=lambda x: np.sqrt(x[0]**2 + x[1]**2),
                         x1=lambda x: np.arctan2(x[1],x[0]))
    c.run_clue()
    c.to_csv('./','circles_output.csv')

    assert cmp('./circles_output.csv', './test_datasets/circles_1000_truth.csv')

if __name__ == "__main__":
    c = clue.clusterer(0.9,5,1.5)
    c.read_data('./test_datasets/circles_1000.csv')
    c.change_coordinates(x0=lambda x: np.sqrt(x[0]**2 + x[1]**2),
                         x1=lambda x: np.arctan2(x[1],x[0]))
    c.run_clue()
    c.cluster_plotter()
