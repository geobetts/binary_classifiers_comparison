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
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.preprocessing import StandardScaler


def knn_prediction_pipeline(train_set,
                            train_targets,
                            test_set,
                            test_targets,
                            kfolds):
    train_set = StandardScaler().fit_transform(train_set)
    test_set = StandardScaler().fit_transform(test_set)

    best_performances = cv.cross_validate_knn(train_set, train_targets, kfolds=kfolds)

    k = random.choice(best_performances)

    neighbours = KNeighborsClassifier(n_neighbors=k)

    knn_fit = neighbours.fit(train_set, train_targets)

    predictions = knn_fit.predict(test_set)

    predictions = predictions.reshape(test_targets.shape)

    return predictions, best_performances, k


def svm_prediction_pipeline(train_set,
                            train_targets,
                            test_set,
                            test_targets,
                            kfolds):
    train_set = StandardScaler().fit_transform(train_set)
    test_set = StandardScaler().fit_transform(test_set)

    best_performances = cv.cross_validate_svm(train_set, train_targets, kfolds=kfolds)

    k = random.choice(best_performances)

    clf = svm.SVC(gamma='auto', kernel=k)

    clf.fit(train_set, train_targets)

    predictions = clf.predict(test_set)

    predictions = predictions.reshape(test_targets.shape)

    return predictions, best_performances, k


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
        predictions, best_performances, k = knn_prediction_pipeline(train_set,
                                                                    train_targets,
                                                                    test_set,
                                                                    test_targets,
                                                                    kfolds)

    if model == "svm":
        predictions, best_performances, k = svm_prediction_pipeline(train_set,
                                                                    train_targets,
                                                                    test_set,
                                                                    test_targets,
                                                                    kfolds)

    accuracy_ratio = pm.accuracy(predicted=predictions, true=test_targets)

    conf_matrix = confusion_matrix(y_true=test_targets, y_pred=predictions, labels=[1, 0])
    precision_value = pm.precision(conf_matrix)
    recall_value = pm.recall(conf_matrix)
    f1 = pm.f1_score(conf_matrix)

    t2 = time.time() - t

    outputs = pd.DataFrame([train_set_name, test_set_name, model, best_performances, k,
                            accuracy_ratio, conf_matrix, precision_value, recall_value, f1, t2])

    outputs = outputs.transpose()

    return outputs
