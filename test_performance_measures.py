"""
Author: G Bettsworth
"""
import numpy as np
import pytest
import performance_measures as pm

generic_confusion = np.array([[2, 4],
                              [8, 6]]
                             )


def test_accuracy():
    output = pm.accuracy(predicted=np.array([[1], [0], [1], [0]]),
                         true=np.array([[1], [1], [1], [0]]))

    assert output == 0.75


def test_precision():
    precision_value = pm.precision(generic_confusion)

    assert precision_value == (2 / (2 + 8))


def test_recall():
    recall_value = pm.recall(generic_confusion)

    assert recall_value == (2 / (2 + 4))


def test_f1():
    f1 = pm.f1_score(generic_confusion)

    assert f1 == 2 * (((2 / (2 + 4)) * (2 / (2 + 8))) / ((2 / (2 + 4)) + (2 / (2 + 8))))
