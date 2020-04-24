"""
Author: G Bettsworth
"""

import numpy as np


def accuracy(predicted, true):
    same_by_row = np.equal(predicted, true)

    accuracy_ratio = sum(same_by_row) / len(true)

    return accuracy_ratio


def recall(confusion_matrix):
    recall_value = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])

    return recall_value


def precision(confusion_matrix):
    precision_value = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])

    return precision_value


def f1_score(confusion_matrix):
    precision_value = precision(confusion_matrix)
    recall_value = recall(confusion_matrix)

    f1 = 2 * ((precision_value * recall_value) / (precision_value + recall_value))

    return f1
