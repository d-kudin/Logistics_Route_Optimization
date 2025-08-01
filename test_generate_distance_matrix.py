# test_generate_distance_matrix.py

import os
import pandas as pd

def test_output_file_exists():
    assert os.path.exists("output/distance_matrix.csv")

def test_matrix_format():
    df = pd.read_csv("output/distance_matrix.csv", index_col=0)
    assert df.shape[0] == df.shape[1]
    assert all(df.columns == df.index)