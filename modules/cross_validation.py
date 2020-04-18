"""
Author: G Bettsworth
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def cross_validate_knn(train_set,
                       train_targets,
                       kfolds,
                       ks_to_test=20
                       ):

    performance = {}

    k_neighbours = range(1, ks_to_test)

    for k in k_neighbours:
        neighbours = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(neighbours, train_set, train_targets, cv=kfolds)
        average = scores.mean()

        performance[k] = average

    maximum = max(performance, key=performance.get)

    best_performance = [k for k, v in performance.items() if float(v) == maximum]

    return best_performance
