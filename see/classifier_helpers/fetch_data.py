"""The purpose of this file is to fetch or create data for benchmarking purposes."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import label_binarize, StandardScaler


def fetch_wisconsin_data():
    """Fetches Breast Cancer Wisconsin (Diagnostic) data online.

    Returns
    -------
    X : array-like of shape (n_samples, n_features)
        The data to fit or predict on where n_samples=569 and n_features=30.

    y : array-like of shape (n_samples,)
        The target label to predict. Labels are binary
        where 1 is Malignant and 0 is Benign

    Notes
    -----
    This function relies on the data found at this url:
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    Data (X) is not preprocessed.

    """

    # Data URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    data = pd.read_csv(url, header=None)

    X = data.iloc[:, 2:].to_numpy()

    y = data[1].to_numpy()
    y = label_binarize(y=y, classes=["B", "M"]).ravel()
    return X, y

def generate_tutorial_data():
    """
    Generates tutorial data.

    Returns
    -------
    datasets : dict
        Dictionary that contains the tutorial datasets.
        Dictionary keys are one of circles, linearly_separable, and moons.
        Dictionary values are tuples (X, y) where:

        X : array-like of shape (n_samples, n_features)
            The data to fit or predict on where n_samples=569 and n_features=30.

        y : array-like of shape (n_samples,)
            The target label to predict. Labels are binary
            where 1 is Malignant and 0 is Benign

    Notes
    -----
    The scikit-learn tutorial that this is function relies on
    is here:
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    Data (X) is preprocessed in the same way that it is done in the 
    tutorial.
    """

    datasets = dict()

    datasets["circles"] = make_circles(noise=0.2, factor=0.5, random_state=1)
    datasets["moons"] = make_moons(noise=0.3, random_state=0)
    X, y = make_classification(
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=1,
            n_clusters_per_class=1,
        )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    datasets["linearly_separable"] = (X, y)

    # Preprocess data
    for name in datasets:
        X, y = datasets[name]
        X = StandardScaler().fit_transform(X)
        datasets[name] = (X, y)

    return datasets
