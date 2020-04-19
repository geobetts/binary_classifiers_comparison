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
    outputs = rp.research_pipeline(train_set=train_set,
                                   train_targets=train_targets,
                                   test_set=test_set,
                                   test_targets=test_targets,
                                   train_set_name='train_set',
                                   test_set_name='test_set',
                                   model="knn")

    print(outputs)
    print("printed outputs work if you're reading this")


def test_pipeline_with_svm():
    outputs = rp.research_pipeline(train_set=train_set,
                                   train_targets=train_targets,
                                   test_set=test_set,
                                   test_targets=test_targets,
                                   train_set_name='train_set',
                                   test_set_name='test_set',
                                   model="svm")

    print(outputs)
    print("printed outputs work if you're reading this")
