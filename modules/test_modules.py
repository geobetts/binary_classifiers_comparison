"""
Author: G Bettsworth
"""

import numpy as np
import pytest
import research_pipeline as rp
import performance_measures as pm

train_set = np.tile(np.array([[1, 2, 3], [3, 2, 1], [1, 2, 3]]), (10, 1))

train_targets = np.tile(np.array([[1], [0], [1]]), (10, 1))

test_set = np.tile(np.array([[1, 2, 3], [1, 2, 3], [3, 2, 1]]), (10, 1))

test_targets = np.tile(np.array([[1], [1], [0]]), (10, 1))

generic_confusion = np.array([[2, 4],
                              [8, 6]]
                             )


def test_pipeline():
    accuracy_ratio, conf_matrix, precision_value, recall_value, f1 = rp.research_pipeline(train_set, train_targets,
                                                                                          test_set, test_targets)

    print(f"accuracy of the model is: {accuracy_ratio}")
    print(conf_matrix)
    print(precision_value)
    print(recall_value)
    print(f1)
    print("printed outputs work if you're reading this")


def test_accuracy():
    output = pm.accuracy(predicted=np.array([[1], [0], [1], [0]]),
                         true=np.array([[1], [1], [1], [0]]))

    assert output == 0.75


def test_precision():
    precision_value = pm.precision(generic_confusion)

    assert precision_value == (2 / (2 + 4))


def test_recall():
    recall_value = pm.recall(generic_confusion)

    assert recall_value == (2 / (2 + 8))


def test_f1():
    f1 = pm.f1_score(generic_confusion)

    assert f1 == 2 * (((2 / (2 + 4)) * (2 / (2 + 8)))/((2 / (2 + 4)) + (2 / (2 + 8))))
