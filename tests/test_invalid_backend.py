import sys
import numpy as np
import pytest
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue

data = np.array([[0., 0., 1.],
                 [0.1, 0., 1.],
                 [0., 0.1, 1.],
                 [5., 5., 1.],
                 [5.1, 5., 1.],
                 [5., 5.1, 1.]])

def test_run_clue_invalid_backend():
    '''
    Test that run_clue raises ValueError for an unrecognised backend name
    '''
    c = clue.clusterer(0.4, 5, 0.4)
    c.read_data(data)
    with pytest.raises(ValueError, match="Invalid backend"):
        c.run_clue(backend="not_a_backend")


def test_fit_invalid_backend():
    '''
    Test that fit raises ValueError for an unrecognised backend name
    '''
    c = clue.clusterer(0.4, 5, 0.4)
    with pytest.raises(ValueError, match="Invalid backend"):
        c.fit(data, backend="not_a_backend")


def test_fit_predict_invalid_backend():
    '''
    Test that fit_predict raises ValueError for an unrecognised backend name
    '''
    c = clue.clusterer(0.4, 5, 0.4)
    with pytest.raises(ValueError, match="Invalid backend"):
        c.fit_predict(data, backend="not_a_backend")


def test_invalid_backend_is_checked_before_reading_data():
    '''
    Test that the backend is validated before the input data is read
    '''
    c = clue.clusterer(0.4, 5, 0.4)
    with pytest.raises(ValueError, match="Invalid backend"):
        c.fit(None, backend="not_a_backend")
    with pytest.raises(ValueError, match="Invalid backend"):
        c.fit_predict(None, backend="not_a_backend")
    with pytest.raises(ValueError, match="Invalid backend"):
        c.run_clue(backend="not_a_backend")
