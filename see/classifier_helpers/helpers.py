import numpy as np
from see.base_classes import pipedata
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

def generate_train_test_set(X, y, random_state=21):
    """
    Split data into training and testing sets
    """
    # Split data into training and testing sets
    dataset = pipedata()
    training_set = pipedata()
    testing_set = pipedata()
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=random_state)

    training_set.X = X_train
    training_set.y = y_train
    testing_set.X = X_test
    testing_set.y = y_test

    dataset.training_set = training_set
    dataset.testing_set = testing_set
    
    dataset.k_folds = False

    return dataset

def generate_tuning_trials(X, y, num_trials=5):
    """
    Use KFolds to split training data into a smaller trials of training and 
    tuning/validation sets to tune the classifier via Genetic Search
    """
    dataset = pipedata()

    kf = KFold(n_splits=num_trials)
    folds = list(kf.split(X))
    n_splits = kf.get_n_splits(X)

    dataset.training_folds = np.empty(n_splits,dtype=object)
    dataset.testing_folds = np.empty(n_splits,dtype=object)

    for i, train_test_index in enumerate(kf.split(X)):
        train_index, test_index = train_test_index
        training_fold = pipedata()
        testing_fold = pipedata()
        training_fold.X, testing_fold.X = X[train_index], X[test_index]
        training_fold.y, testing_fold.y = y[train_index], y[test_index]
        dataset.training_folds[i] = training_fold
        dataset.testing_folds[i] = testing_fold

    dataset.k_folds = True
    return dataset