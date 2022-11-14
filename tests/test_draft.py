import pandas as pd
import pytest

def test_always_passes():
    assert True

@pytest.fixture
def blobs():
    return pd.read_csv("./blob.csv")

import CLUEstering as clue
def test_correct_number_of_blobs(blobs):
    c = clue.clusterer(1,5,1.6)
    c.readData(blobs)
    c.runCLUE()
    assert c.NClusters == 4
