import pandas as pd
import pytest
import time
import CLUEstering as clue

@pytest.fixture
def blobs():
    return pd.read_csv("./blob.csv")

def test_time(blobs):
    c = clue.clusterer(1,5,1.6)
    c.readData(blobs)
    start = time.time_ns()
    c.runCLUE()
    finish = time.time_ns()

    elapsed_time = (finish - start)/(10**9)
    assert elapsed_time < 1.0641621732029838
