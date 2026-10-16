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

def test_run_clue_backend_not_compiled(monkeypatch):
    '''
    Test that run_clue raises RuntimeError when the requested backend
    is valid but the library was not compiled with support for it
    '''
    c = clue.clusterer(0.4, 5, 0.4)
    c.read_data(data)
    monkeypatch.setattr(clue, 'tbb_found', False)
    with pytest.raises(RuntimeError, match="not available"):
        c.run_clue(backend="cpu tbb")


def test_fit_backend_not_compiled(monkeypatch):
    '''
    Test that fit raises RuntimeError when the requested backend
    is valid but the library was not compiled with support for it
    '''
    c = clue.clusterer(0.4, 5, 0.4)
    monkeypatch.setattr(clue, 'omp_found', False)
    with pytest.raises(RuntimeError, match="not available"):
        c.fit(data, backend="cpu openmp")


def test_fit_predict_backend_not_compiled(monkeypatch):
    '''
    Test that fit_predict raises RuntimeError when the requested backend
    is valid but the library was not compiled with support for it
    '''
    c = clue.clusterer(0.4, 5, 0.4)
    monkeypatch.setattr(clue, 'cuda_found', False)
    with pytest.raises(RuntimeError, match="not available"):
        c.fit_predict(data, backend="gpu cuda")


def test_list_devices_still_prints_for_uncompiled_backend(capsys):
    '''
    Test that list_devices keeps printing an informative message
    instead of raising when a backend is not compiled
    '''
    c = clue.clusterer(0.4, 5, 0.4)
    if not clue.is_tbb_available():
        c.list_devices('cpu tbb')
        assert "not found" in capsys.readouterr().out
    if not clue.is_openmp_available():
        c.list_devices('cpu openmp')
        assert "not found" in capsys.readouterr().out
