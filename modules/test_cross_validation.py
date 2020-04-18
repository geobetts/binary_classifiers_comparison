"""
Author: G Bettsworth
"""

import numpy as np
import pytest
import cross_validation as cv

train_set = np.tile(np.array([[1, 2, 3], [3, 2, 1], [1, 2, 3]]), (10, 1))

train_targets = np.tile(np.array([[1], [0], [1]]), (10, 1))


def test_cv():
    scores = cv.cross_validate_knn(train_set, train_targets, 5)

    print(f"best_value_is k = {scores}")
