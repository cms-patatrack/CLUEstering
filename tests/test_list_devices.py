'''
Test the method for querying available devices
'''

import sys
import pandas as pd
import pytest
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue

def test_list_devices():
    '''
    Test the method for querying available devices
    '''
    c = clue.clusterer(0.4, 5, 0.4)

    c.list_devices()
    c.list_devices('cpu serial')
    c.list_devices('cpu tbb')
    c.list_devices('cpu openmp')
    c.list_devices('gpu cuda')
    c.list_devices('gpu hip')
    assert True


def test_list_devices_invalid_backend():
    '''
    Test that list_devices raises ValueError for an unrecognised backend name
    '''
    c = clue.clusterer(0.4, 5, 0.4)
    with pytest.raises(ValueError):
        c.list_devices('not_a_backend')
