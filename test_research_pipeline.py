"""
Author: G Bettsworth
"""

import numpy as np
import pytest
import research_pipeline as rp

train_set = np.tile(np.array([[1, 2, 3], [3, 2, 1], [1, 2, 3]]), (10, 1))

train_targets = np.tile(np.array([[1], [0], [1]]), (10, 1))

test_set = np.tile(np.array([[1, 2, 3], [1, 2, 3], [3, 2, 1]]), (10, 1))

test_targets = np.tile(np.array([[1], [1], [0]]), (10, 1))


def test_pipeline_with_knn():
    accuracy_ratio, conf_matrix, precision_value, recall_value, f1 = rp.research_pipeline(train_set,
                                                                                          train_targets,
                                                                                          test_set,
                                                                                          test_targets,
                                                                                          knn=True,
                                                                                          svm=False)

    print(f"accuracy of the model is: {accuracy_ratio}")
    print(conf_matrix)
    print(precision_value)
    print(recall_value)
    print(f1)
    print("printed outputs work if you're reading this")


def test_pipeline_with_svm():
    accuracy_ratio, conf_matrix, precision_value, recall_value, f1 = rp.research_pipeline(train_set,
                                                                                          train_targets,
                                                                                          test_set,
                                                                                          test_targets,
                                                                                          knn=False,
                                                                                          svm=True)

    print(f"accuracy of the model is: {accuracy_ratio}")
    print(conf_matrix)
    print(precision_value)
    print(recall_value)
    print(f1)
    print("printed outputs work if you're reading this")
