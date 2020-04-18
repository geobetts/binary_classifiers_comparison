"""
Author: G Bettsworth
"""

import numpy as np
import performance_measures as pm
import cross_validation as cv
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def research_pipeline(train_set,
                      train_targets,
                      test_set,
                      test_targets,
                      kfolds=5
                      ):

    best_performances = cv.cross_validate_knn(train_set, train_targets, kfolds=kfolds)

    k = random.choice(best_performances)

    neighbours = KNeighborsClassifier(n_neighbors=k)

    knn_fit = neighbours.fit(train_set, train_targets)

    predictions = knn_fit.predict(test_set)

    predictions = predictions.reshape(test_targets.shape)

    accuracy_ratio = pm.accuracy(predicted=predictions, true=test_targets)

    conf_matrix = confusion_matrix(y_true=test_targets, y_pred=predictions, labels=[1, 0])
    precision_value = pm.precision(conf_matrix)
    recall_value = pm.recall(conf_matrix)
    f1 = pm.f1_score(conf_matrix)

    return accuracy_ratio, conf_matrix, precision_value, recall_value, f1
