"""This runs unit tests for functions that can be found in classifiers.py."""

import numpy as np

from sklearn.datasets import make_moons  # , make_circles, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as DecisionTree

from sklearn.model_selection import train_test_split

from see import classifiers


def test_gaussian_naive_bayes():
    """Unit test for Gaussian Naive Bayes classifer algorithm.
    Check if evaluate function output is same as it would be
    running the sklearn function manually."""

    # Generate dataset
    h = 0.02

    dataset = make_moons(noise=0.3, random_state=0)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf = GaussianNB()

    clf.fit(X_train, y_train)  # Train classifier

    # manual sklearn categorizations
    expected_predictions = clf.predict_proba(
        np.c_[xx.ravel(), yy.ravel()])[:, 1]
    expected_predictions = expected_predictions.reshape(xx.shape)

    # manual sklearn fitness score
    expected_score = clf.score(X_test, y_test)

    # see-classify categorizations
    see_clf = classifiers.GaussianNBClassifier()
    [actual_predictions, actual_score] = see_clf.evaluate(dataset)

    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions == expected_predictions).all()
    assert actual_score == expected_score


def test_nearest_neighbor():
    """Unit test for Nearest Neighbors classifier algorithm.
    Check if evaluate function output is same as it would be
    running the sklearn function manually."""

    # Generate dataset
    h = 0.02

    dataset = make_moons(noise=0.3, random_state=0)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf = KNeighborsClassifier(3)

    clf.fit(X_train, y_train)  # Train classifier

    # manual sklearn categorizations
    expected_predictions = clf.predict_proba(
        np.c_[xx.ravel(), yy.ravel()])[:, 1]
    expected_predictions = expected_predictions.reshape(xx.shape)

    # manual sklearn fitness score
    expected_score = clf.score(X_test, y_test)

    # see-classify categorizations
    see_clf = classifiers.KNeighborsClassifier()
    [actual_predictions, actual_score] = see_clf.evaluate(dataset)

    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions == expected_predictions).all()
    assert actual_score == expected_score


def test_decision_tree():
    """Unit test for Decision Tree classifier algorithm.
    Check if evaluate function output is same as it would be
    running the sklearn function manually."""

    # Generate dataset
    h = 0.02

    dataset = make_moons(noise=0.3, random_state=0)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf = DecisionTree(max_depth=5)

    clf.fit(X_train, y_train)  # Train classifier

    # manual sklearn categorizations
    expected_predictions = clf.predict_proba(
        np.c_[xx.ravel(), yy.ravel()])[:, 1]
    expected_predictions = expected_predictions.reshape(xx.shape)

    # manual sklearn fitness score
    expected_score = clf.score(X_test, y_test)

    # see-classify categorizations
    see_clf = classifiers.DecisionTreeClassifier()
    [actual_predictions, actual_score] = see_clf.evaluate(dataset)

    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions == expected_predictions).all()
    assert actual_score == expected_score

# TODO: This test fails. I think it is because the algorithm
# has a randomness to it.


def test_random_forest():
    """Unit test for Decision Tree classifier algorithm.
    Check if evaluate function output is same as it would be
    running the sklearn function manually."""

    # Generate dataset
    h = 0.02

    dataset = make_moons(noise=0.3, random_state=0)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

    clf.fit(X_train, y_train)  # Train classifier

    # manual sklearn categorizations
    expected_predictions = clf.predict_proba(
        np.c_[xx.ravel(), yy.ravel()])[:, 1]
    expected_predictions = expected_predictions.reshape(xx.shape)

    # manual sklearn fitness score
    expected_score = clf.score(X_test, y_test)

    # see-classify categorizations
    see_clf = classifiers.RandomForestContainer()
    [actual_predictions, actual_score] = see_clf.evaluate(dataset)

    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions == expected_predictions).all()
    assert actual_score == expected_score

# TODO: This test fails. I think it is because the algorithm
# has a randomness to it.


def test_mlp():
    """Unit test for Decision Tree classifier algorithm.
    Check if evaluate function output is same as it would be
    running the sklearn function manually."""

    # Generate dataset
    h = 0.02

    dataset = make_moons(noise=0.3, random_state=0)
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    clf = MLPClassifier(alpha=1, max_iter=1000)

    clf.fit(X_train, y_train)  # Train classifier

    # manual sklearn categorizations
    expected_predictions = clf.predict_proba(
        np.c_[xx.ravel(), yy.ravel()])[:, 1]
    expected_predictions = expected_predictions.reshape(xx.shape)

    # manual sklearn fitness score
    expected_score = clf.score(X_test, y_test)

    # see-classify categorizations
    see_clf = classifiers.MLPContainer()
    [actual_predictions, actual_score] = see_clf.evaluate(dataset)

    assert len(actual_predictions) == len(expected_predictions)
    assert (actual_predictions == expected_predictions).all()
    assert actual_score == expected_score
