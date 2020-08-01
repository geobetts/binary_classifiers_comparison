"""
Author: G Bettsworth
"""

import numpy as np
import pandas as pd
import performance_measures as pm
import cross_validation as cv
import random
import time
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn import svm
from sklearn.preprocessing import StandardScaler


def knn_prediction_pipeline(train_set,
                            train_targets,
                            test_set,
                            test_targets,
                            kfolds):

    train_set = StandardScaler().fit_transform(train_set)
    test_set = StandardScaler().fit_transform(test_set)

    best_performances, maximum = cv.cross_validate_knn(train_set, train_targets, kfolds=kfolds)

    random.seed(123)

    k = random.choice(best_performances)

    neighbours = KNeighborsClassifier(n_neighbors=k[0], metric=k[1], weights=k[2], leaf_size=k[3])

    knn_fit = neighbours.fit(train_set, train_targets)

    predictions = knn_fit.predict(test_set)

    predictions = predictions.reshape(test_targets.shape)

    return predictions, best_performances, k, maximum


def svm_prediction_pipeline(train_set,
                            train_targets,
                            test_set,
                            test_targets,
                            kfolds):

    train_set = StandardScaler().fit_transform(train_set)
    test_set = StandardScaler().fit_transform(test_set)

    best_performances, maximum = cv.cross_validate_svm(train_set, train_targets, kfolds=kfolds)

    random.seed(123)

    k = random.choice(best_performances)

    clf = svm.SVC(C=k[0], gamma=k[1], kernel=k[2], decision_function_shape=k[3], random_state=123)

    clf.fit(train_set, train_targets)

    predictions = clf.predict(test_set)

    predictions = predictions.reshape(test_targets.shape)

    return predictions, best_performances, k, maximum


def research_pipeline(train_set,
                      train_set_name,
                      train_targets,
                      test_set,
                      test_set_name,
                      test_targets,
                      model,
                      kfolds=5
                      ):
    t = time.time()

    if model == "knn":
        predictions, best_performances, k, maximum = knn_prediction_pipeline(train_set,
                                                                    train_targets,
                                                                    test_set,
                                                                    test_targets,
                                                                    kfolds)

    if model == "svm":
        predictions, best_performances, k, maximum = svm_prediction_pipeline(train_set,
                                                                    train_targets,
                                                                    test_set,
                                                                    test_targets,
                                                                    kfolds)

    accuracy_ratio = sklearn.metrics.accuracy_score(y_true=test_targets, y_pred=predictions)

    conf_matrix = sklearn.metrics.confusion_matrix(y_true=test_targets, y_pred=predictions, labels=[1, 0])
    precision_value = sklearn.metrics.precision_score(y_true=test_targets, y_pred=predictions)
    recall_value = sklearn.metrics.recall_score(y_true=test_targets, y_pred=predictions)
    f1 = sklearn.metrics.f1_score(y_true=test_targets, y_pred=predictions)

    t2 = time.time() - t

    outputs = pd.DataFrame([train_set_name, test_set_name, model, best_performances, k, maximum,
                            accuracy_ratio, conf_matrix, precision_value, recall_value, f1, t2])

    outputs = outputs.transpose()

    return outputs
