"""
Author: G Bettsworth
"""

import numpy as np
import pytest
import research_pipeline as rp

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

train_set = np.tile(np.array([[1, 2, 3], [3, 2, 1], [1, 2, 3]]), (10, 1))

train_targets = np.tile(np.array([[1], [0], [1]]), (10, 1))

test_set = np.tile(np.array([[1, 2, 3], [1, 2, 3], [3, 2, 1]]), (10, 1))

test_targets = np.tile(np.array([[1], [1], [0]]), (10, 1))


def test_pipeline_with_knn():
    outputs = rp.prediction_pipeline(train_set=train_set,
                                train_targets=train_targets,
                                test_set=test_set,
                                test_targets=test_targets,
                                model=KNeighborsClassifier())
    
    print(outputs)
    print("printed outputs work if you're reading this")


def test_pipeline_with_svm():
    outputs = rp.prediction_pipeline(train_set=train_set,
                                train_targets=train_targets,
                                test_set=test_set,
                                test_targets=test_targets,
                                model=SVC())
    
    print(outputs)
    print("printed outputs work if you're reading this")
