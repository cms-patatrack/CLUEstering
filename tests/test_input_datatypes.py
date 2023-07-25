import numpy as np
import pandas as pd
import pytest
import CLUEstering as clue

@pytest.fixture
def file():
    csv_file = './test_datasets/blob.csv'
    return csv_file

@pytest.fixture
def dataframe():
    csv_file = './test_datasets/blob.csv'
    df_ = pd.read_csv(csv_file)
    return df_

@pytest.fixture
def dictionary(dataframe):
    data_dict = {'x0': dataframe['x0'].values.tolist(),
                 'x1': dataframe['x1'].values.tolist(),
                 'x2': dataframe['x2'].values.tolist(),
                 'weight': dataframe['weight'].values.tolist()}
    return data_dict

@pytest.fixture
def lists(dataframe):
    data_lists = [dataframe['x0'].values.tolist(),
                  dataframe['x1'].values.tolist(),
                  dataframe['x2'].values.tolist(),
                  dataframe['weight'].values.tolist()]
    return data_lists

@pytest.fixture
def arrays(dataframe):
    data_arrays = np.array([np.array(dataframe['x0'].values.tolist()),
                            np.array(dataframe['x1'].values.tolist()),
                            np.array(dataframe['x2'].values.tolist()),
                            np.array(dataframe['weight'].values.tolist())])
    return data_arrays

# Test the different data types singularly, so to make it easier to debug in case of error
def test_csv(file):
    """
    Test that CLUE works when the data is written in a csv file.
    """

    clust = clue.clusterer(1,5,1.6)
    clust.read_data(file)
    clust.run_clue()

def test_dict(dictionary):
    """
    Test that CLUE works when the data is contained in a dictionary.
    """

    clust = clue.clusterer(1,5,1.6)
    clust.read_data(dictionary)
    clust.run_clue()

def test_pddf(dataframe):
    """
    Test that CLUE works when the data is contained in a pandas dataframe.
    """

    clust = clue.clusterer(1,5,1.6)
    clust.read_data(dataframe)
    clust.run_clue()

def test_list(lists):
    """
    Test that CLUE works when the data is contained in lists.
    """

    clust = clue.clusterer(1,5,1.6)
    clust.read_data(lists)
    clust.run_clue()

def test_ndarray(arrays):
    """
    Test that CLUE works when the data is contained in numpy ndarrays.
    """

    clust = clue.clusterer(1,5,1.6)
    clust.read_data(arrays)
    clust.run_clue()

def test_same_result(file, dictionary, dataframe, lists, arrays):
    """
    Run CLUE for all the supported data types and assert that the output is the same
    for all of them.
    """

    clust_file = clue.clusterer(1,5,1.6)
    clust_file.read_data(file)
    clust_file.run_clue()

    clust_dict = clue.clusterer(1,5,1.6)
    clust_dict.read_data(dictionary)
    clust_dict.run_clue()

    clust_df = clue.clusterer(1,5,1.6)
    clust_df.read_data(dataframe)
    clust_df.run_clue()

    clust_list = clue.clusterer(1,5,1.6)
    clust_list.read_data(lists)
    clust_list.run_clue()

    clust_arr = clue.clusterer(1,5,1.6)
    clust_arr.read_data(arrays)
    clust_arr.run_clue()

    assert clust_file.clust_prop == clust_dict.clust_prop
    assert clust_dict.clust_prop == clust_df.clust_prop
    assert clust_df.clust_prop == clust_list.clust_prop
    assert clust_list.clust_prop == clust_arr.clust_prop
