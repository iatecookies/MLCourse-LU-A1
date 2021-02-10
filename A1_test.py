import numpy as np


import A1


# Do NOT change this file


def test_training_test_split():
    # A dataset that makes it easy to see if we're splitting X and y in the same way
    X = np.arange(30)
    X = X.reshape((10, 3), order='F')
    y = np.arange(10)

    X_train, X_test, y_train, y_test = A1.training_test_split(X, y, test_size=0.4)  # 40% test set
    print('X_train[:,0]'.ljust(15), X_train[:, 0])
    print('y_train'.ljust(15), y_train)  # you should be able to see if the X and y columns align correctly

    # Test if the correct data type is returned.
    assert type(X_train) is np.ndarray, "X_train isn't a np.ndarray"
    assert type(y_train) is np.ndarray, "y_train isn't a np.ndarray"

    # Test if the correct shape is returned.
    assert X_test.shape == (4, 3), "Test set did not get exactly 40% of the instances."
    assert X_train.shape == (6, 3), "Training set did not get exactly 100 - 40 = 60% of the instances."

    # Test that the training and test sets were properly separated
    # Since each column in the original data had only unique values, there should be no shared values after the split
    for i in range(X_train.shape[1]):  # columns 0..n
        assert len(np.intersect1d(X_train[:, i], X_test[:, i])) == 0, \
            f"X_train and X_test have shared instances in column {i}"

    # Test that for the class labels as well.
    assert len(np.intersect1d(y_train, y_test)) == 0, \
        f"y_train and y_test have shared instances: \n{y_train}\n{y_test}"

    # Test if the instances in the training set are correctly aligned.
    assert np.array_equal(X_train[:, 0], y_train)

    # Test if the instances in the test set are correctly aligned.
    assert np.array_equal(X_test[:, 0], y_test)

    # Test if setting the random seed works
    X_train_1, _, _, _ = A1.training_test_split(X, y, random_state=42)
    X_train_2, _, _, _ = A1.training_test_split(X, y, random_state=42)
    assert np.array_equal(X_train_1, X_train_2), "Using the same random seed should result in identical splits"

    # Test if NOT setting the random seed works
    # We do this three times,
    # because it's unlikely that we shuffle the same way three times
    X_train_1, _, _, _ = A1.training_test_split(X, y)
    X_train_2, _, _, _ = A1.training_test_split(X, y)
    X_train_3, _, _, _ = A1.training_test_split(X, y)
    assert not (np.array_equal(X_train_1, X_train_2) and
                np.array_equal(X_train_1, X_train_3) and
                np.array_equal(X_train_2, X_train_3)
                ), "If the random seed is not set, results should not be reproducible."

    # Test if the training set is actually shuffled
    # We do this three times, because it's realy really unlikely,
    # that we would happen to get a sorted array randomly three times in a row.

    X_train_1, _, _, _ = A1.training_test_split(X, y)  # Note that the random seed is NOT set!
    col0 = X_train_1[:, 0]  # select column 0
    sorting_1 = col0.argsort()  # indices of column 0 sorted by values
    X_train_1_sorted = X_train_1[sorting_1]  # now sort the array by those sorted indices

    X_train_2, _, _, _ = A1.training_test_split(X, y)
    X_train_2_sorted = X_train_2[X_train_2[:, 0].argsort()]  # the same thing but as a one-liner

    X_train_3, _, _, _ = A1.training_test_split(X, y)
    X_train_3_sorted = X_train_3[X_train_3[:, 0].argsort()]

    assert not (np.array_equal(X_train_1, X_train_1_sorted) and
                np.array_equal(X_train_2, X_train_2_sorted) and
                np.array_equal(X_train_3, X_train_3_sorted)
                ), "X_train should not be sorted"


def test_negatives():
    #                 [ TP,  TP,  FN,  TN,  TN,  TN,  FP   FN   FN   FN]
    y_test = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'A'])
    y_pred = np.array(['A', 'A', 'B', 'B', 'B', 'B', 'A', 'B', 'B', 'B'])
    assert A1.true_negatives(y_test, y_pred, 'A') == 3
    assert A1.false_negatives(y_test, y_pred, 'A') == 4


def test_recall():
    # Data with       [ TP,  TP,  FN,  FP,  TN,  FP,  FP]
    y_pred = np.array(['A', 'A', 'B', 'A', 'B', 'A', 'A'])
    y_test = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'B'])

    assert A1.recall(y_test, y_pred, 'A') == 2 / (2 + 1)  # TP=2, FN=1


def test_accuracy():
    # Data with       [ TP,  TP,  FN,  FP,  TN,  FP,  FP]
    y_pred = np.array(['A', 'A', 'B', 'A', 'B', 'A', 'A'])
    y_test = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'B'])

    assert A1.accuracy(y_test, y_pred, 'A') == (2 + 1) / (2 + 1 + 1 + 3)  # TP=2, TN=1, FN=1, FP=3


def test_specificity():
    # Data with       [ TP,  TP,  TP,  TP,  TP,  FP,  FP]
    y_pred = np.array(['A', 'A', 'A', 'A', 'A', 'A', 'A'])
    y_test = np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B'])

    assert A1.specificity(y_test, y_pred, 'A') == 0 / (0 + 1)  # TP=0, FP=1


def test_balanced_accuracy():
    # Data with       [ TP,  TP,  TP,  TP,  TP,  FP,  FP]
    y_pred = np.array(['A', 'A', 'A', 'A', 'A', 'A', 'A'])
    y_test = np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B'])

    assert A1.balanced_accuracy(y_test, y_pred, 'A') == 0.5  # (recall=1 + specificity=0) / 2


def test_F1():
    # Data with       [ TP,  TP,  TP,  TP,  TP,  FP,  FP]
    y_pred = np.array(['A', 'A', 'A', 'A', 'A', 'A', 'A'])
    y_test = np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B'])

    assert 0.83 < A1.F1(y_test, y_pred, 'A') < 0.84
