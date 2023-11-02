from filecmp import cmp
import CLUEstering as clue
import numpy as np
import os
import pandas as pd
import pytest
import sys
sys.path.insert(1, '../CLUEstering/')


@pytest.fixture
def circles():
    return pd.read_csv("./test_datasets/circles.csv")


def test_circles_clustering(circles):
    # Check if the output file already exists and if it does, delete it
    if os.path.isfile('./circles_output.csv'):
        os.remove('./circles_output.csv')

    c = clue.clusterer(0.9, 5, 1.5)
    c.read_data(circles)
    c.change_coordinates(x0=lambda x: np.sqrt(x[0]**2 + x[1]**2),
                         x1=lambda x: np.arctan2(x[1], x[0]))
    c.run_clue()
    c.to_csv('./', 'circles_output.csv')

    assert cmp('./circles_output.csv',
               './test_datasets/truth_files/circles_1000_truth.csv')
