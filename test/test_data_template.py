import sys
sys.path.append('./bin')
sys.path.append('../bin')
def test_demo():
    assert 100 == 100

import pytest
import pandas as pd
from v_data_template import DataBaseLoader

def db_loader():
    return DataBaseLoader()

def test_load_files_positive():
    db_loader = DataBaseLoader()
    id = "id"
    data1 = pd.DataFrame({'id': [1, 2, 3], 'col1': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'id': [1, 2, 3], 'col2': [10, 20, 30]})
    expected_output = pd.DataFrame({'id': [1, 2, 3], 'col1': ['a', 'b', 'c'], 'col2': [10, 20, 30]})
    output = db_loader.load_data_frame(id, data1, data2)
    assert output.equals(expected_output)

def test_load_files_negative():
    db_loader = DataBaseLoader()
    id = "id"
    data1 = pd.DataFrame({'id': [1, 2, 3], 'col1': ['a', 'b', 'c']})
    data2 = pd.DataFrame({'id': [1, 2, 4], 'col2': [10, 20, 30]})
    expected_output = pd.DataFrame({'id': [1, 2], 'col1': ['a', 'b'], 'col2': [10, 20]})
    output = db_loader.load_data_frame(id, data1, data2)
    assert output.equals(expected_output)
