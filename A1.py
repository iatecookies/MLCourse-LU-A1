import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, ensemble, metrics, svm, model_selection, linear_model


def training_test_split(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    """ Split the features X and labels y into training and test features and labels. 

    `split` indicates the fraction (rounded down) that should go to the test set.

    `random_state` allows to set a random seed to make the split reproducible. 
    If `random_state` is None, then no random seed will be set.

    """
    raise NotImplementedError('Your code here')
    return X_train, X_test, y_train, y_test


def true_positives(true_labels, predicted_labels, positive_class):
    pos_true = true_labels == positive_class  # compare each true label with the positive class
    pos_predicted = predicted_labels == positive_class # compare each predicted label to the positive class
    match = pos_true & pos_predicted # use logical AND (that's the `&`) to find elements that are True in both arrays
    return np.sum(match)  # count them


def false_positives(true_labels, predicted_labels, positive_class):
    pos_predicted = predicted_labels == positive_class  # predicted to be positive class
    neg_true = true_labels != positive_class  # actually negative class
    match = pos_predicted & neg_true  # The `&` is element-wise logical AND
    return np.sum(match)  # count the number of matches


def true_negatives(true_labels, predicted_labels, positive_class):
    raise NotImplementedError('Your code here')


def false_negatives(true_labels, predicted_labels, positive_class):
    raise NotImplementedError('Your code here')


def precision(true_labels, predicted_labels, positive_class):
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FP = false_positives(true_labels, predicted_labels, positive_class)
    return TP / (TP + FP)


def recall(true_labels, predicted_labels, positive_class):
    raise NotImplementedError('Your code here')


def accuracy(true_labels, predicted_labels, positive_class):
    raise NotImplementedError('Your code here')


def specificity(true_labels, predicted_labels, positive_class):
    raise NotImplementedError('Your code here')


def balanced_accuracy(true_labels, predicted_labels, positive_class):
    raise NotImplementedError('Your code here')


def F1(true_labels, predicted_labels, positive_class):
    raise NotImplementedError('Your code here')


def load_data(fraction=0.75, seed=None, target_digit=9, appply_stratification=True):
    data = sklearn.datasets.load_digits()
    X = data.data
    y = data.target
    y[y != target_digit] = 11  # we have to do this swap because 1 and 0 also occur as labels in our dataset
    y[y == target_digit] = 12
    y[y == 11] = 0  # negative class
    y[y == 12] = 1  # positive class
    if appply_stratification:
        stratify = y
    else:
        stratify = None
    return sklearn.model_selection.train_test_split(X, y, train_size=fraction, random_state=seed, stratify=stratify)