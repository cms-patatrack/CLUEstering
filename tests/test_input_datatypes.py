'''
Testing the algorithm with all the supported input data types
'''

import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


def test_read_array_except():
    '''
    Test the exception raised when passing incorrect arrays
    '''
    arr = np.array([[1, 4, 5]])
    clust = clue.clusterer(0.4, 5., 1.2)

    with pytest.raises(ValueError):
        clust.read_data(arr)


def test_read_string_except():
    '''
    Test the exception raised when passing incorrect data files
    '''
    clust = clue.clusterer(0.4, 5., 1.2)

    with pytest.raises(ValueError):
        clust.read_data('./test_datasets/blob.dat')


@pytest.fixture
def no_weight_dataset():
    '''
    Returns a dataset with no weight associated to the points
    '''
    x0 = np.array([0, 1, 2, 3, 4])
    x1 = np.array([5, 6, 7, 8, 9])
    x2 = np.array([10, 11, 12, 13, 14])
    data = {'x0': x0, 'x1': x1, 'x2': x2}

    return pd.DataFrame(data)


@pytest.fixture
def low_dimensionality_dataset():
    '''
    Returns a dataset with no coordinates
    '''
    weight = np.array([1, 1, 1, 1, 1])
    data = {'weight': weight}

    return pd.DataFrame(data)


@pytest.fixture
def high_dimensionality_dataset():
    '''
    Returns a 11-dimensional dataset
    '''
    x0 = np.array([0, 1, 2, 3, 4])
    x1 = np.array([0, 1, 2, 3, 4])
    x2 = np.array([0, 1, 2, 3, 4])
    x3 = np.array([0, 1, 2, 3, 4])
    x4 = np.array([0, 1, 2, 3, 4])
    x5 = np.array([0, 1, 2, 3, 4])
    x6 = np.array([0, 1, 2, 3, 4])
    x7 = np.array([0, 1, 2, 3, 4])
    x8 = np.array([0, 1, 2, 3, 4])
    x9 = np.array([0, 1, 2, 3, 4])
    x10 = np.array([0, 1, 2, 3, 4])
    weight = np.array([1, 1, 1, 1, 1])
    data = {'x0': x0, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
            'x6': x6, 'x7': x7, 'x8': x8, 'x9': x9, 'x10': x10, 'weight': weight}

    return pd.DataFrame(data)


def test_handle_dataframe_except(no_weight_dataset,
                                 low_dimensionality_dataset,
                                 high_dimensionality_dataset):
    '''
    Test the error handling when passing incorrect dataframes
    '''
    clust = clue.clusterer(0.5, 5., 1.)

    with pytest.raises(ValueError):
        clust._handle_dataframe(no_weight_dataset)
    with pytest.raises(ValueError):
        clust._handle_dataframe(low_dimensionality_dataset)
    with pytest.raises(ValueError):
        clust._handle_dataframe(high_dimensionality_dataset)


@pytest.fixture
def file():
    '''
    Returns the path to a test csv file
    '''
    csv_file = './test_datasets/blob.csv'
    return csv_file


@pytest.fixture
def dataframe():
    '''
    Returns the dataframe of a test dataset
    '''
    csv_file = './test_datasets/blob.csv'
    df_ = pd.read_csv(csv_file)
    return df_


@pytest.fixture
def dictionary(dataframe):
    '''
    Returns a test dataset as dictionary
    '''
    data_dict = {'x0': dataframe['x0'].values.tolist(),
                 'x1': dataframe['x1'].values.tolist(),
                 'x2': dataframe['x2'].values.tolist(),
                 'weight': dataframe['weight'].values.tolist()}
    return data_dict


@pytest.fixture
def lists(dataframe):
    '''
    Returns a test dataset as a list of lists
    '''
    data_lists = [dataframe['x0'].values.tolist(),
                  dataframe['x1'].values.tolist(),
                  dataframe['x2'].values.tolist(),
                  dataframe['weight'].values.tolist()]
    return data_lists


@pytest.fixture
def arrays(dataframe):
    '''
    Returns a test dataset as an array of arrays
    '''
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

    clust = clue.clusterer(1, 5, 1.6)
    clust.read_data(file)
    clust.run_clue()


def test_dict(dictionary):
    """
    Test that CLUE works when the data is contained in a dictionary.
    """

    clust = clue.clusterer(1, 5, 1.6)
    clust.read_data(dictionary)
    clust.run_clue()


def test_pddf(dataframe):
    """
    Test that CLUE works when the data is contained in a pandas dataframe.
    """

    clust = clue.clusterer(1, 5, 1.6)
    clust.read_data(dataframe)
    clust.run_clue()


def test_list(lists):
    """
    Test that CLUE works when the data is contained in lists.
    """

    clust = clue.clusterer(1, 5, 1.6)
    clust.read_data(lists)
    clust.run_clue()


def test_ndarray(arrays):
    """
    Test that CLUE works when the data is contained in numpy ndarrays.
    """

    clust = clue.clusterer(1, 5, 1.6)
    clust.read_data(arrays)
    clust.run_clue()


def test_same_result(file, dictionary, dataframe, lists, arrays):
    """
    Run CLUE for all the supported data types and assert that the output is the
    same for all of them.
    """

    clust_file = clue.clusterer(1, 5, 1.6)
    clust_file.read_data(file)
    clust_file.run_clue()

    clust_dict = clue.clusterer(1, 5, 1.6)
    clust_dict.read_data(dictionary)
    clust_dict.run_clue()

    clust_df = clue.clusterer(1, 5, 1.6)
    clust_df.read_data(dataframe)
    clust_df.run_clue()

    clust_list = clue.clusterer(1, 5, 1.6)
    clust_list.read_data(lists)
    clust_list.run_clue()

    clust_arr = clue.clusterer(1, 5, 1.6)
    clust_arr.read_data(arrays)
    clust_arr.run_clue()

    assert clust_file.clust_prop == clust_dict.clust_prop
    assert clust_dict.clust_prop == clust_df.clust_prop
    assert clust_df.clust_prop == clust_list.clust_prop
    assert clust_list.clust_prop == clust_arr.clust_prop
