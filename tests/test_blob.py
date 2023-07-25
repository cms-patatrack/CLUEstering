import pandas as pd
import pytest
import CLUEstering as clue

@pytest.fixture
def blobs():
    return pd.read_csv("./test_datasets/blob.csv")

def test_correct_number_of_blobs(blobs):
    c = clue.clusterer(1,5,1.6)
    c.read_data(blobs)
    c.run_clue()
    c.to_csv('./','file.csv')
