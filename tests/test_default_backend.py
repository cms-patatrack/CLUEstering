'''
Test that run_clue, fit and fit_predict work when no backend is specified
'''

import sys
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue

_preference_order = ["gpu cuda", "gpu hip", "cpu tbb", "cpu openmp", "cpu serial"]


def _expected_default_backend() -> str:
    '''
    Returns the first compiled backend in the order of preference
    '''
    return next(backend for backend in _preference_order if backend in clue.backends)


def _chosen_backend(output: str) -> str:
    '''
    Extracts the backend name from the verbose output of run_clue
    '''
    prefix = 'Running CLUEstering with backend: '
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1
    return lines[0][len(prefix):]


@pytest.fixture
def blobs():
    '''
    Returns the dataframe containing the blob dataset
    '''
    return pd.read_csv("../data/blob.csv")


def test_run_clue_default_backend(blobs, capsys):
    '''
    Test that run_clue without a backend uses the preferred available one
    '''
    c = clue.clusterer(1., 5, 2.)
    c.read_data(blobs)
    c.run_clue(verbose=True)

    assert _chosen_backend(capsys.readouterr().out) == _expected_default_backend()
    mask = c.cluster_ids != -1
    assert silhouette_score(c.coords.T[mask], c.cluster_ids[mask]) > 0.8


def test_fit_default_backend(blobs, capsys):
    '''
    Test that fit without a backend uses the preferred available one
    '''
    c = clue.clusterer(1., 5, 2.)
    assert c.fit(blobs, verbose=True) is c

    assert _chosen_backend(capsys.readouterr().out) == _expected_default_backend()
    mask = c.cluster_ids != -1
    assert silhouette_score(c.coords.T[mask], c.cluster_ids[mask]) > 0.8


def test_fit_predict_default_backend(blobs, capsys):
    '''
    Test that fit_predict without a backend uses the preferred available one
    '''
    c = clue.clusterer(1., 5, 2.)
    labels = c.fit_predict(blobs, verbose=True)

    assert _chosen_backend(capsys.readouterr().out) == _expected_default_backend()
    assert (labels == c.cluster_ids).all()
    mask = labels != -1
    assert silhouette_score(c.coords.T[mask], labels[mask]) > 0.8
