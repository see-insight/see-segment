"""This runs unit tests for functions that can be found in classifiers.py."""

import numpy as np

from see import classifiers
from sklearn.datasets import make_moons #, make_circles, make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def test_gaussian_naive_bayes():
    """Unit test for GNB. Check if evaluate function output is same as
    it would be running the sklearn function manually."""

    # Generate dataset
    h = 0.02

    dataset = make_moons(noise=0.3, random_state=0)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf = GaussianNB()

    clf.fit(X, y)  # Train classifier

    # sklearn categorizations
    expected_predictions = clf.predict_proba(
        np.c_[xx.ravel(), yy.ravel()])[:, 1]
    expected_predictions = expected_predictions.reshape(xx.shape)

    # see-classify categorizations
    see_clf = classifiers.NaiveBayes()
    actual_predictions = see_clf.evaluate(dataset)

    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions == expected_predictions).all()
