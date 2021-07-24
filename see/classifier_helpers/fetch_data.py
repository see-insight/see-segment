# fetch_data.py
"""The purpose of this file is to fetch data for benchmarking purposes."""
import pandas as pd
import numpy as np
import urllib
from sklearn.preprocessing import label_binarize

def fetch_wisconsin_data():
    """Fetches Breast Cancer Wisconsin (Diagnostic) data online.
    Returns:
    X, feature vectors
    y, labels where 1 is Malignant and 0 is Benign
    """
    # Breast Cancer Data URL
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
    data = pd.read_csv(url, header=None)

    X = data.iloc[:,2:].to_numpy()

    y = data[1].to_numpy()
    y = label_binarize(y=y,classes=['B', 'M']).ravel()
    return X, y