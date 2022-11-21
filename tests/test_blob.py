import pandas as pd
import pytest
import CLUEstering as clue

@pytest.fixture
def blobs():
    return pd.read_csv("blob.csv")

def test_correct_number_of_blobs(blobs):
    c = clue.clusterer(1,5,1.6)
    c.readData(blobs)
    c.runCLUE()
    assert c.NClusters == 4
