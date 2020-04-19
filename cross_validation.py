"""
Author: G Bettsworth
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import itertools


def cross_validate_knn(train_set,
                       train_targets,
                       kfolds,
                       ks_to_test=20
                       ):

    print("Cross Validation:")

    performance = {}

    k_neighbours = range(1, ks_to_test)

    distances = ["euclidean", "manhattan", "chebyshev", "minkowski"]

    weights = ["uniform", "distance"]

    # leaf size doesn't seem to affect quality
    leaf_size = [30]

    params = [k_neighbours, distances, weights, leaf_size]

    all_variations = list(itertools.product(*params))

    for k in all_variations:
        neighbours = KNeighborsClassifier(n_neighbors=k[0], metric=k[1], weights=k[2], leaf_size=k[3])
        scores = cross_val_score(neighbours, train_set, train_targets, cv=kfolds)
        average = scores.mean()

        performance[k] = average

        print(f"{k}. Score: {average}")

    maximum = max(performance.values())

    best_performance = [k for k, v in performance.items() if float(v) == maximum]

    return best_performance, maximum


def cross_validate_svm(train_set,
                       train_targets,
                       kfolds
                       ):

    print("Cross Validation:")

    performance = {}

    C = list(range(1, 10))

    gamma = ["scale", "auto"]

    kernels = ["linear", "poly", "rbf", "sigmoid"]

    # decision function shape doesn't seem to make a difference

    decision_function_shape = ["ovr"]

    params = [C, gamma, kernels, decision_function_shape]

    all_variations = list(itertools.product(*params))

    for k in all_variations:
        clf = svm.SVC(C=k[0], gamma=k[1], kernel=k[2], decision_function_shape=k[3], random_state=123)
        scores = cross_val_score(clf, train_set, train_targets, cv=kfolds)
        average = scores.mean()

        performance[k] = average

        print(f"{k}. Score: {average}")

    maximum = max(performance.values())

    best_performance = [k for k, v in performance.items() if float(v) == maximum]

    return best_performance, maximum
